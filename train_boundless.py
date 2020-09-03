# Train the boundless model on a set of images 
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_value_

import torchvision

import argparse

import os
import glob
import numpy as np
from PIL import Image

import models.losses.synthesis as synthesis


def hinge_loss(fake, real, is_generator):
    if is_generator:
        loss = - (fake.mean(dim=1).mean())
    else:
        real_loss = F.relu(torch.ones(real.size()).to(real.device) - real)
        fake_loss = F.relu(torch.ones(fake.size()).to(fake.device) + fake)
        loss = real_loss.mean(dim=1).mean() + fake_loss.mean(dim=1).mean()

    return loss 

class GatedConv(nn.Module):
    def __init__(self, in_c, in_o, k, s, d):
        super().__init__()
        self.gating = nn.Conv2d(in_c, in_o, kernel_size=k, stride=s, dilation=d, padding=d*(k//2))
        self.weight = nn.Conv2d(in_c, in_o, kernel_size=k, stride=s, dilation=d, padding=d*(k//2))
        self.norm = nn.InstanceNorm2d(in_o)

    def forward(self, input):
        feats = F.elu(self.weight(input))
        gates = F.sigmoid(self.gating(input))

        return self.norm(feats * gates)

class Dataset(data.Dataset):
    def __init__(self, images_str, masks_str, is_training):

        super().__init__()
        ims = glob.glob(images_str)
        masks = glob.glob(masks_str)

        self.ims = ims
        self.masks = masks
        self.training = is_training

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((350,350)),
            torchvision.transforms.ToTensor()]
        )

        self.normalize = torchvision.transforms.Normalize(
            (0.5,0.5,0.5), (0.5,0.5,0.5))


    def __len__(self):
        return 2 ** 32

    def __getitem__(self, index):
        # index = 0
        rng = np.random.RandomState(index)
        im_id = rng.randint(len(self.ims))
        mask_id = rng.randint(len(self.masks))

        im = Image.open(self.ims[im_id])
        mask = Image.open(self.masks[mask_id])
        

        if self.training:
            im = np.array(im)
            
            # Crop the image
            o_x = rng.randint(0.1 * im.shape[1])
            o_y = rng.randint(0.1 * im.shape[0])

            min_h = int(0.8 * im.shape[0])
            h = rng.randint(im.shape[0] - o_y - min_h) + min_h
            min_w = int(0.8 * im.shape[1])
            w = rng.randint(im.shape[1] - o_x - min_w) + min_w

            im = im[o_y:o_y+h,o_x:o_x+w,:]
            im = Image.fromarray(im)

        im = self.normalize(self.transform(im))

        mask = (self.transform(mask)[0:1,:,:] > 0).float()
        crop = im * (1 - mask)

        return {'Im' : im, 'Mask' : mask, 'Crop' : crop}

class ValDataset(data.Dataset):
    def __init__(self, images_str, masks_str, is_training):

        super().__init__()
        ims = sorted(glob.glob(images_str))
        masks = sorted(glob.glob(masks_str))

        self.base_impath = ims[0].split('-')[0]
        self.base_maskpath = masks[0].split('-')[0]
        self.num_ims = len(ims)

        self.ims = ims
        self.masks = masks
        self.training = is_training

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((350,350)),
            torchvision.transforms.ToTensor()]
        )

        self.normalize = torchvision.transforms.Normalize(
            (0.5,0.5,0.5), (0.5,0.5,0.5))


    def __len__(self):
        return self.num_ims

    def __getitem__(self, index):
        im = Image.open(self.base_impath + '-%03d.png' % index)
        mask = Image.open(self.base_maskpath + '-%03d.png' % index)

        im = self.normalize(self.transform(im))

        mask = (self.transform(mask)[0:1,:,:] > 0).float()
        crop = im * (1 - mask)

        # import pdb; pdb.set_trace()

        return {'Im' : im, 'Mask' : mask, 'Crop' : crop}


class DNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=5, stride=2, padding=2), nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2), nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2), nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2), nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2), nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2), nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2), nn.LeakyReLU()
        )

        self.fc = nn.Linear(2304,256)

    def forward(self, im):
        B = im.size(0)
        feats = self.layers(im)

        return self.fc(feats.view(B,-1))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.incep_v3 = torchvision.models.inception_v3(pretrained=True)
        self.incep_v3.requires_grad = False
        self.dNet = DNet()

        self.fc_fN = nn.Linear(256, 1)
        self.fc_fC = nn.Linear(1000, 256)

    def forward(self, real, fake, mask):
        # Compute values
        with torch.no_grad():
            fs_inp = self.incep_v3(real)[0]
        ad_inp = self.dNet(torch.cat((fake, mask), 1))

        # Normalize
        fs_inp = fs_inp / torch.norm(fs_inp, 1)

        fN = self.fc_fN(ad_inp)
        fC = self.fc_fC(fs_inp) 

        pred = fN + (fC * ad_inp).sum(dim=1, keepdim=True)

        return pred

class BoundlessModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Construct the model
        self.conv1 = GatedConv(  4,  32, 5, 1, 1)
        self.conv2 = GatedConv( 32,  64, 3, 2, 1)
        self.conv3 = GatedConv( 64,  64, 3, 1, 1)
        self.conv4 = GatedConv( 64, 128, 3, 2, 1)
        self.conv5 = GatedConv(128, 128, 3, 1, 1)
        self.conv6 = GatedConv(128, 128, 3, 1, 1)
        self.conv7 = GatedConv(128, 128, 3, 1, 2)
        self.conv8 = GatedConv(128, 128, 3, 1, 4)
        self.conv9 = GatedConv(128, 128, 3, 1, 8)
        self.conv10 = GatedConv(128, 128, 3, 1, 16)
        self.conv11 = GatedConv(128, 128, 3, 1, 1)
        self.conv12 = GatedConv(256, 128, 3, 1, 1)
        self.conv14 = GatedConv(256, 64, 3, 1, 1)
        self.conv15 = GatedConv(128, 64, 3, 1, 1)
        self.conv17 = GatedConv(128, 32, 3, 1, 1)
        self.conv18 = GatedConv(64, 16, 3, 1, 1)
        self.conv19 = nn.Conv2d(16, 3, kernel_size=3, stride=1, dilation=1)

    def forward(self, im):
        im_c1 = self.conv1(im)
        im_c2 = self.conv2(im_c1)
        im_c3 = self.conv3(im_c2)
        im_c4 = self.conv4(im_c3)
        im_c5 = self.conv5(im_c4)
        im_c6 = self.conv6(im_c5)
        im_c7 = self.conv7(im_c6)
        im_c8 = self.conv8(im_c7)
        im_c9 = self.conv9(im_c8)
        im_c10 = self.conv10(im_c9)

        im_c11 = torch.cat((self.conv11(im_c10), im_c5), 1)
        im_c12 =  torch.cat((self.conv12(im_c11), im_c4), 1)
        im_c13 = F.upsample(size=im_c3.size()[2:], input=im_c12, mode='bilinear')
        im_c14 =  torch.cat((self.conv14(im_c13), im_c3), 1)
        im_c15 =  torch.cat((self.conv15(im_c14), im_c2), 1)
        im_c16 = F.upsample(size=im_c1.size()[2:], input=im_c15, mode='bilinear')
        im_c17 =  torch.cat((self.conv17(im_c16), im_c1), 1)

        im_c18 = self.conv18(im_c17)
        im_c19 = self.conv19(im_c18)

        pred_im = F.upsample(im_c19, size=im.size()[2:], mode='bilinear')
        pred_im = F.tanh(pred_im)
        return pred_im

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = Discriminator()
        self.model = BoundlessModel()

        self.optimizerG = torch.optim.Adam(
                    list(self.model.parameters()),
                    lr=0.0001,
                    betas=(0.5, 0.9)
                )
        self.optimizerD = torch.optim.Adam(
                    list(self.discriminator.parameters()),
                    lr=0.001,
                    betas=(0.5,0.9)
                )

        self.init_weights(init_type='normal')
    
    def init_weights(self, gain=0.02, init_type='xavier'):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, "weight") and (
                (classname.find("Conv") != -1 and classname.find("GatedConv") != 0) or classname.find("Linear") != -1
            ):
                if init_type == "normal":
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == "xavier":
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == "xavier_uniform":
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == "kaiming":
                    init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                elif init_type == "orthogonal":
                    init.orthogonal_(m.weight.data)
                elif init_type == "":  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        "initialization method [%s] is not implemented"
                        % self.opt.init
                    )
                if hasattr(m, "bias") and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, "init_weights"):
                m.init_weights(self.opt.init, gain)

    def __call__(self, iter_data_loader, isval=False, n_iters=8, use_adv=False):
        if isval:
            with torch.no_grad():
                pred = self.model(torch.cat((batch['Crop'], batch['Mask']),1))
                pred = pred * batch['Mask'] + (1 - batch['Mask']) * batch['Crop']
                real = batch['Im']
                l1_loss = synthesis.L1LossMask()(pred, real, batch['Mask'])
            return {'Pred' : pred, 'Im': batch['Im'], 'Crop' : batch['Crop'], 
                    'Mask' : batch['Mask']}, {'L1Loss' : l1_loss}
        else:
            if use_adv:
                reals = []; preds = []; masks = []

            self.optimizerG.zero_grad()
            for i in range(0, n_iters):
                batch = iter_data_loader.next()
                batch = {k : batch[k].cuda() for k in batch.keys()}
                pred = self.model(torch.cat((batch['Crop'], batch['Mask']),1))
                pred = pred * batch['Mask'] + (1 - batch['Mask']) * batch['Crop']
                real = batch['Im']

                l1_loss = synthesis.L1LossMask()(pred, real, batch['Mask'])
                (l1_loss / float(n_iters)).backward(retain_graph=use_adv)

                if use_adv:
                    reals += [real.detach()]
                    preds += [pred.detach()]
                    masks += [batch['Mask']]
                    f_adv = self.discriminator(real, pred, batch['Mask'])
                    adv_loss = hinge_loss(pred, None, True)      
                    (0.01 * adv_loss / float(n_iters)).backward()   

            # clip_grad_value_(self.model.parameters(), 0.001)
   
            self.optimizerG.step()

            if use_adv:
                self.optimizerD.zero_grad()
                for i in range(0, n_iters):
                    real_f = self.discriminator(reals[i], reals[i], masks[i])
                    fake_f = self.discriminator(reals[i], preds[i], masks[i])
                    loss = hinge_loss(fake_f, real_f, False)
                    (0.01 * loss / float(n_iters)).backward()

                # clip_grad_value_(self.discriminator.parameters(), 0.001)
                
                self.optimizerD.step()

            if use_adv:
                results =  {'L1Loss' : l1_loss, 'G' : adv_loss, 'D' : loss}
            else:
                results = {'L1Loss' : l1_loss}
            return {'Pred' : pred, 'Im': batch['Im'], 'Crop' : batch['Crop'], 
                    'Mask' : batch['Mask']}, results

    def get_optimizer(self):
        return {'OptimG': self.optimizerG, 'OptimD' : self.optimizerD}

def train(opts):
    # Create the dataset
    dataset = Dataset(opts.image, opts.mask, opts.training)
    data_loader = data.DataLoader(
        dataset=dataset,
        num_workers=4,
        batch_size=opts.batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )
    iter_data_loader = iter(data_loader)

    # Create the model
    model = BaseModel()
    model = model.cuda()

    if not os.path.exists('%s/' % opts.model_path):
        os.makedirs('%s/' % opts.model_path)
        os.makedirs('%s/images/' % opts.model_path)
        os.makedirs('%s/checkpoints/' % opts.model_path)

    # Then look at training
    for i in range(0, opts.n_iters):

        ims, results = model(iter_data_loader)

        # And then write the images and loss
        if i % 100 == 0:
            torchvision.utils.save_image(ims['Pred'] * 0.5 + 0.5, 
                                   '%s/images/pred%d.png' % (opts.model_path, i))
            torchvision.utils.save_image(ims['Im'] * 0.5 + 0.5, 
                                   '%s/images/real%d.png' % (opts.model_path, i))
            torchvision.utils.save_image(ims['Mask'] * 0.5 + 0.5, 
                                   '%s/images/mask%d.png' % (opts.model_path, i))
            torchvision.utils.save_image(ims['Crop'] * 0.5 + 0.5, 
                                   '%s/images/crop%d.png' % (opts.model_path, i))
        if i % 10 == 0:
            s = 'Errors: Iter %d -' % i 
            for k in results.keys():
                s += ' %s | %0.4f; ' % (k, results[k])
            print(s)

        # And every so often write the model checkpoint
        if i % 100 == 0:
            checkpoint_state = {
                "state_dict": model.state_dict(),
                "optimizerG": model.optimizerG.state_dict(),
                "optimizerD": model.optimizerD.state_dict(),
                "opts": opts,
                }

            torch.save(checkpoint_state, '%s/checkpoints/model_iter%d.pth' % (opts.model_path, i))

def evaluate(opts):
    if not os.path.exists('%s' % opts.val_path):
        os.makedirs('%s' % opts.val_path)

    dataset = ValDataset(opts.image, opts.mask, opts.training)
    data_loader = data.DataLoader(dataset=dataset, num_workers=0, batch_size=1, 
                                  shuffle=False, drop_last=True, pin_memory=True)

    iter_data_loader = iter(data_loader)

    # Load the model again
    model = BaseModel()
    model = model.cuda()

    # Load the state dict
    state_dict = torch.load(opts.model_path)
    model.load_state_dict(state_dict['state_dict'])

    # And now let's go through all the images + masks and generate them
    batch = next(iter_data_loader, None)
    i = 0
    while batch:
        print("Iters %d of %d" % (i, len(dataset)))
        pred = model.model(torch.cat((batch['Crop'].cuda(), batch['Mask'].cuda()),1))
        pred = pred * batch['Mask'].cuda() + (1 - batch['Mask'].cuda()) * batch['Crop'].cuda()

        torchvision.utils.save_image(pred * 0.5 + 0.5, 
                                '%s/pred%03d.png' % (opts.val_path, i))
        torchvision.utils.save_image(batch['Im'] * 0.5 + 0.5, 
                                '%s/real%03d.png' % (opts.val_path, i))
        torchvision.utils.save_image(batch['Mask'] * 0.5 + 0.5, 
                                '%s/mask%03d.png' % (opts.val_path, i))
        torchvision.utils.save_image(batch['Crop'] * 0.5 + 0.5, 
                                '%s/crop%03d.png' % (opts.val_path, i))

        batch = next(iter_data_loader, None)
        i += 1


if __name__ == '__main__':
    arguments = argparse.ArgumentParser()
    arguments.add_argument('--n_iters', type=int)
    arguments.add_argument('--mask', type=str, default='000010')
    arguments.add_argument('--image', type=str)
    arguments.add_argument('--val_path', type=str, default='')
    arguments.add_argument('--model_path', type=str, default='000010')
    arguments.add_argument('--batch_size', type=int, default=16)
    arguments.add_argument('--training', action='store_true')

    opts = arguments.parse_args()

    if opts.training:
        train(opts)
    else:
        evaluate(opts)

    