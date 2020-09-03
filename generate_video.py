# Import pytorch 3d

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import compositing
from pytorch3d.renderer.points import rasterize_points

import numpy as np
from PIL import Image
from tqdm import tqdm

import cv2

import matplotlib.pyplot as plt

# Now use interpolation in order to generate intermediary views
import torch
import torchvision.transforms as tr
import torch.nn.functional as F

import os
import torchvision

import argparse

def generate_video(im_name, normalise):
    # Load in the correspondences and images
    im1 = Image.open(os.environ['BASE_PATH'] + '/imL/%s.jpg' % im_name)
    im2 = Image.open(os.environ['BASE_PATH'] + '/imR/%s.jpg' % im_name)

    if normalise:
        np_tempwarp = np.load(
            os.environ['BASE_PATH'] + '/warps/temp_sampler%s_2_grad_norm.npz' % im_name)
        H = np_tempwarp['H']
        np_tempwarp = np_tempwarp['sampler']
    else:
        np_tempwarp = np.load(
            os.environ['BASE_PATH'] + '/warps/temp_sampler%s_2_grad_coarse.npz.npy' % im_name)

    if normalise:
        im1_arr = np.array(im1)
        im2_arr = np.array(im2)

        K1 = np.eye(3)
        K1[0,0] = 2 / im1_arr.shape[1]; K1[1,1] = 2 / im1_arr.shape[0] 
        K1[0:2,2] = - 1 
        K2 = np.eye(3)
        K2[0,0] = 2 / im2_arr.shape[1]; K2[1,1] = 2 / im2_arr.shape[0] 
        K2[0:2,2] = - 1 

        aff_mat = (np.linalg.inv(K2) @ H @ K1)

        # Now transform the image and return
        warp_im1 = cv2.warpAffine(im1_arr, aff_mat[0:2], (im2_arr.shape[1], im2_arr.shape[0]))

        im1 = warp_im1
        im1 = Image.fromarray(im1)

    warp = torch.Tensor(np_tempwarp).unsqueeze(0)

    im1_torch = tr.ToTensor()(im1).unsqueeze(0)
    im2_torch = tr.ToTensor()(im2).unsqueeze(0)

    gen_img = F.grid_sample(im1_torch, warp)

    sampler = F.upsample(warp.permute(0,3,1,2), size=(im2_torch.size(2), im2_torch.size(3)))
    gen_imglarge = F.grid_sample(im1_torch, sampler.permute(0,2,3,1))

    W1, W2, _ = np_tempwarp.shape
    orig_warp = torch.meshgrid(torch.linspace(-1,1,W1), torch.linspace(-1,1,W2))
    orig_warp = torch.cat((orig_warp[1].unsqueeze(2), orig_warp[0].unsqueeze(2)), 2)

    orig_warp = orig_warp.unsqueeze(0)
    warp = torch.Tensor(np_tempwarp).unsqueeze(0)

    new_imgs = []

    if not os.path.exists('./temp%s/%s' % (im_name, im_name)):
        os.makedirs('./temp%s/%s' % (im_name, im_name))

    radius = 2 * 2 / 1024.
        
    for i in tqdm(range(-10, 30)):
        resample = (orig_warp * float(i) / 20. + warp * float(20 - i) / 20.)
        pts3D = resample.view(1,-1,2)
        
        pts_mask = (warp.view(-1,2)[:,0] > -1) & (warp.view(-1,2)[:,0] < 1)
        
        pts3D = pts3D[:,pts_mask,:]
        pts3D = - pts3D
        pts3D = torch.cat((pts3D.cuda(), torch.ones((1,pts3D.size(1),1)).cuda()), 2)
        
        rgb = F.grid_sample(im2_torch, orig_warp).permute(0,2,3,1).view(1,-1,3)[:,pts_mask,:]
        mask = torch.ones((1,rgb.size(1),1)).cuda()
        
        pts3DRGB = Pointclouds(points=pts3D, features=rgb)
        points_idx, _, dist = rasterize_points(pts3DRGB, 1024, radius, 1)
        gen_img = pts3DRGB.features_packed()[points_idx.permute(0,3,1,2).long()[0],:].permute(0,3,1,2).mean(dim=0, keepdim=True)
        new_imgs += [gen_img.squeeze().permute(1,2,0)]
        
        torchvision.utils.save_image(gen_img, './temp%s/%s/im-%03d.png' % (im_name, im_name, i+10))
        
        mask = (points_idx.permute(0,3,1,2) < 0).float()
        
        torchvision.utils.save_image(mask, 
                                    './temp%s/%s/mask-%03d.png' % (im_name, im_name, i+10))


if __name__ == '__main__':
    arguments = argparse.ArgumentParser()
    arguments.add_argument('--im_name', type=str, default='000010')
    arguments.add_argument('--normalise', action='store_true')

    args = arguments.parse_args()

    generate_video(args.im_name, args.normalise)