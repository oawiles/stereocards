import requests
import wget
import os

import glob

import cv2
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms

import time

from tqdm import tqdm

def best_matches(sim, cond1=None, cond2=None, topk=8000, T=0.3, nn=1):
    ''' Find the best matches for a given NxN matrix.
        Optionally, pass in the actual indices corresponding to this matrix
        (cond1, cond2) and update the matches according to these indices.
    '''
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]

    ids1 = torch.arange(0, sim.shape[0]).to(nn12.device)
    mask = (ids1 == nn21[nn12])

    matches = torch.stack([ids1[mask], nn12[mask]])

    preds = sim[ids1[mask], nn12[mask]]
    res, ids = preds.sort()
    ids = ids[res > T]

    if not(cond1 is None) and not(cond2 is None):
        cond_ids1 = cond1[ids1[mask]]
        cond_ids2 = cond2[nn12[mask]]

        matches = torch.stack([cond_ids1, cond_ids2])

    matches = matches[:,ids[-topk:]]

    return matches.t(), None

class GetMatches:
    def __init__(self, model, transform):
        self.model = model
        self.transform = transform
        
    def iterate(self, im1_orig, im2_orig, iters=1):
        trans_imgs = [np.array(im1_orig.resize(im2_orig.size))]
        Hs = []
        
        im1 = self.transform(im1_orig)
        im2 = self.transform(im2_orig)
            
        W1, H1 = im1_orig.size
        W2, H2 = im2_orig.size

        # Now visualise heat map for a given point
        with torch.no_grad():
            results = self.model.run_feature((im1.unsqueeze(0).cuda(), None), (im2.unsqueeze(0).cuda(), None), 
                MAX=16384, keypoints=None, sz1=(128,128), sz2=(128,128), 
                factor=1, r_im1=(1,1), r_im2=(1,1), 
                use_conf=1, T=0.1, 
                return_4dtensor=True, RETURN_ALLKEYPOINTS=True)

        matches = results['match']
        
        # find the match over the second set of keypoints
        prob_match, ids_matches = matches.view(128*128,128*128).max(dim=1)

        kps1 = results['kp1'].clone()
        kps1[:,0] = (results['kp1'][:,0] / (1 * 2.) + 0.5) * W1
        kps1[:,1] = (results['kp1'][:,1] / (1 * 2.) + 0.5) * H1

        kps2 = results['kp2'].clone()
        kps2[:,0] = (results['kp2'][:,0] / (1 * 2.)  + 0.5) * W2
        kps2[:,1] = (results['kp2'][:,1] / (1 * 2.)  + 0.5) * H2
        
        return kps1, kps2[ids_matches,:], prob_match

def download_text():
    
    with open(os.environ['BASE_PATH'] + '/nypl_large_names.txt', 'r') as f:
        lines = f.read().splitlines()
        last_line = lines[-1]
    
    last_id = int(last_line.split('_')[0])

    fi_name = open(os.environ['BASE_PATH'] + '/nypl_large_names.txt', 'a+')
    while True:
        try:
            # First get the collection uuid
            print("Obtaining collection...")
            url = 'http://api.repo.nypl.org/api/v1/collections'
            base_name = os.environ['BASE_PATH'] + '/nypl_large/'
            auth = 'Token token=XXXXXXXX'
            call = requests.get(url, headers={'Authorization': auth})

            collections = call.json()['nyplAPI']['response']['collection']

            for collection in tqdm(collections):
                if 'stereoscopic' in collection['title']:
                    uuid = collection['uuid']
                    print('UuID : %s' % uuid)

                    print('NumItems : %s' % collection['numItems'])
                    break

            # Then get each element and start downloading!
            print('Starting download...')
            url = 'http://api.repo.nypl.org/api/v1/items/%s' % uuid
            call = requests.get(url, headers={'Authorization': auth})

            response = call.json()['nyplAPI']['response']
            numItems = int(response['numResults'])

            # numPagination = len(response['collection'])
            i = 1

            print("Testing...")
            while os.path.exists('%s/%06d.tiff' % (base_name, i * 10 - 1)) and ((i - 1) * 10 < last_id):
                print("Skipping %d" % i)
                i += 1

            offset = (i - 1) * 10
            print(offset)

            # Download Item
            print("Downloading individual elements...")
            item_url = 'http://api.repo.nypl.org/api/v1/items/%s' % uuid
            call = requests.get(item_url, headers={'Authorization': auth}, params={'page' : i})

            # And extract images
            item_response = call.json()['nyplAPI']['response']

            while offset < int(numItems):
                print('Num Results: ', int(numItems))
                for j in tqdm(range(0, len(item_response['capture']), 1)):
                    if item_response['capture'][j]['imageLinks'] is None:
                        offset += 1
                        continue

                    t_uid = item_response['capture'][j]['uuid']
                    print(offset, j, t_uid)
                    item_url_title = 'http://api.repo.nypl.org/api/v1/items/mods_captures/%s' % t_uid
                    call_title = requests.get(item_url_title, headers={'Authorization': auth})
                    try:
                        title = call_title.json()['nyplAPI']['response']['mods']['titleInfo']
                        titles = []
                        if isinstance(title, list):
                            for t_id in range(0, len(title)):
                                titles += [title[t_id]['title']['$']]
                        else:
                                titles += [title['title']['$']]
                    except:
                        pass
                    print(titles)
                    fi_name.write('%06d_' % (offset))
                    for title in titles:
                        fi_name.write('%s_' % title)
                    fi_name.write('\n')
                    offset += 1

                fi_name.flush()

                i += 1
                call = requests.get(item_url, headers={'Authorization': auth}, params={'page' : i})
                item_response = call.json()['nyplAPI']['response']

            if offset == int(numItems):
                break

        except Exception as e:
            print(e)
            # Wait 5 min and try again
            # time.sleep(60 * 5)

    fi_name.close()

def download():

    while True:
        try:
            # First get the collection uuid
            print("Obtaining collection...")
            url = 'http://api.repo.nypl.org/api/v1/collections'
            base_name = os.environ['BASE_PATH'] + '/nypl_large/'
            auth = 'Token token=ab2y0h7wu11hmj2z'
            call = requests.get(url, headers={'Authorization': auth})

            collections = call.json()['nyplAPI']['response']['collection']

            for collection in tqdm(collections):
                if 'stereoscopic' in collection['title']:
                    uuid = collection['uuid']
                    print('UuID : %s' % uuid)

                    print('NumItems : %s' % collection['numItems'])
                    break

            # Then get each element and start downloading!
            print('Starting download...')
            url = 'http://api.repo.nypl.org/api/v1/items/%s' % uuid
            call = requests.get(url, headers={'Authorization': auth})

            response = call.json()['nyplAPI']['response']
            numItems = int(response['numResults'])

            # numPagination = len(response['collection'])
            i = 1

            print("Testing...")
            while os.path.exists('%s/%06d.jpg' % (base_name, i * 10 - 1)):
                print("Skipping %d" % i)
                i += 1

            offset = (i - 1) * 10

            # Download Item
            print("Downloading individual elements...")
            item_url = 'http://api.repo.nypl.org/api/v1/items/%s' % uuid
            call = requests.get(item_url, headers={'Authorization': auth}, params={'page' : i})

            # And extract images
            item_response = call.json()['nyplAPI']['response']

            while offset < int(numItems):
                print('Num Results: ', int(numItems))
                for j in tqdm(range(0, len(item_response['capture']), 1)):
                    if item_response['capture'][j]['imageLinks'] is None:
                        offset += 1
                        continue

                    capture_link = item_response['capture'][j]['highResLink']
                    name = '%06d.tiff' % offset

                    if not(os.path.exists(os.environ['BASE_PATH'] + '/nypl_large/%s' % name)):
                        try:
                            wget.download(capture_link, os.environ['BASE_PATH'] + '/nypl_large/%s' % name)
                        except:
                            pass
                    offset += 1

                i += 1
                call = requests.get(item_url, headers={'Authorization': auth}, params={'page' : i})
                item_response = call.json()['nyplAPI']['response']

            if offset == int(numItems):
                break

        except Exception as e:
            print(e)
            # Wait 5 min and try again
            # time.sleep(60 * 5)

def get_xs(im):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)

    # Do line detection on the image
    lines = cv2.HoughLines(edges,1,np.pi/180,120)

    xlines = None
    for rho,theta in lines[:,0,:]:
        if theta < 0.03:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            x1 = int(x0 + 1000*(-b))
            x2 = int(x0 - 1000*(-b))

            new_x = (x1 + x2) / 2.
            
            if xlines is None:
                xlines = np.array([new_x])
            else:
                xlines = np.hstack((xlines, new_x))


    if xlines is None:
        raise Exception("No x lines")
    indices = np.argsort(xlines)
    xlines = np.array([xlines[i] for i in indices])

    h, w, _ = im.shape

    # And now remove all lines within 10 pixels
    # Middle line should be between [350,400]
    mid_x = np.where((xlines > 350) & (xlines < 400))[0]

    if mid_x.shape[0] < 1:
        raise Exception("No middle line found")

    mid_x = xlines[mid_x].mean()

    # Other lines should be between [0,100] and [-100,:]
    # They should be roughly equivalent
    l_x = np.where((xlines < 100))[0]
    l_x = xlines[l_x]

    r_x = np.where((xlines > w - 100))[0]
    r_x = xlines[r_x]

    # And finally choose the first one that has a line within 10 pixels the other side
    d_l = mid_x - l_x
    d_r = r_x - mid_x
    for i in range(0, len(r_x)):
        t_d = d_r[i]
        
        diff = np.where(np.abs(t_d - d_l) < 20)[0]
        if diff.shape[0] >= 1:
            r_x = r_x[i:i+1]
            l_x = l_x[diff[-1]:diff[-1]+1]
            
            break

    if (l_x.shape[0] > 1) or (r_x.shape[0] > 1) or (l_x.shape[0] == 0) or (r_x.shape[0] == 0):
        raise Exception("No proper edges found")

    return int(l_x[0]), int(r_x[0]), int(mid_x)

def get_ys(im):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)

    # Do line detection on the image
    lines = cv2.HoughLines(edges,1,np.pi/180,140)

    ylines = None
    for rho,theta in lines[:,0,:]:
        if theta < np.pi / 2. + 0.02 and theta > np.pi / 2. - 0.02:
            a = np.cos(theta)
            b = np.sin(theta)
            y0 = b*rho
            y1 = int(y0 + 1000*(a))
            y2 = int(y0 - 1000*(a))

            new_y = (y1 + y2) / 2.
            
            if ylines is None:
                ylines = np.array([new_y])
            else:
                ylines = np.hstack((ylines, new_y))


    indices = np.argsort(ylines)
    ylines = np.array([ylines[i] for i in indices])

    min_y = np.where(ylines < 30)[0]
    if min_y.shape[0] < 1:
        min_y = 30
    else:
        min_y = ylines[min_y[-1]]

    max_y = np.where(ylines > 340)[0]
    if max_y.shape[0] < 1:
        max_y = 350
    else:
        max_y = ylines[max_y[0]]

    h, w, _ = im.shape

    return int(min_y), int(max_y)

def cut_up_images():
    images = sorted(os.listdir(os.environ['BASE_PATH'] + '/nypl_large/'))

    base_dir = os.environ['BASE_PATH'] + '/nypl_preprocessed/'

    preprocessed_ims = sorted(os.listdir(base_dir + 'imR'))
    ind = images.index(preprocessed_ims[-1])

    if not os.path.exists(base_dir):
        os.makedirs(base_dir + 'imR')
        os.makedirs(base_dir + 'imL')

    for im_file in images[ind:]:
        if os.path.exists(base_dir + 'imR/%s' % im_file):
            print("Skipping %s" % im_file)
            continue

        print(im_file)

        # Load in the image
        im = np.array(Image.open(os.environ['BASE_PATH'] + '/nypl_large/' + im_file))
        try:
            l_x, r_x, mid_x = get_xs(im)
            y_min, y_max = get_ys(im)
        except Exception as e:
            if e.args[0] == 'No middle line found':
                print("No middle line found for %s" % im_file)
                continue
            elif e.args[0] == 'No proper edges found':
                print("No proper edges found for %s" % im_file)
                continue
            elif e.args[0] == 'No x lines':
                print("No x lines found for %s" % im_file)
                continue
            else:
                raise e

        # And show the images divided by these x values
        im_l = im[y_min:y_max,l_x:mid_x,:]
        im_r = im[y_min:y_max,mid_x:r_x,:]

        if mid_x == r_x or l_x == mid_x:
            continue

        # Then save these images
        im_l = Image.fromarray(im_l).save(
            base_dir + 'imL/%s' % im_file)
        im_r = Image.fromarray(im_r).save(
            base_dir + 'imR/%s' % im_file)

def find_keypoints():
    base_path = os.environ['BASE_PATH'] + '/nypl_large_preprocessed'
    images = sorted(os.listdir(base_path + '/imR/'))

    if not os.path.exists(base_path + '/keypts/'):
        os.makedirs(base_path + '/keypts/')

    file_name = glob.glob('/scratch/local/hdd/ow/saved_models/d2d/effnetB1_ep86.pth')[0]

    old_model = torch.load(file_name)

    # Load the dataset and model
    opts = old_model['opts']
    opts.W = 512

    model = get_model(opts)
    model.load_state_dict(old_model['state_dict'])

    model = model.eval()

    model = model.cuda()

    npzfiles = sorted(os.listdir(base_path + '/keypts/'))
    last_img = npzfiles[-1][:-4] + '.jpg'
    img_index = images.index(last_img)

    # Consider two specific images
    for img in tqdm(images[img_index:]):
        im1 = os.environ['BASE_PATH'] + '/nypl_large_preprocessed/imL/%s' % img
        im2 = os.environ['BASE_PATH'] + '/nypl_large_preprocessed/imR/%s' % img

        # Load in two images
        transform = transforms.Compose([
            transforms.Resize((opts.W,opts.W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        im1_orig = Image.open(im1).convert("RGB")
        im2_orig = Image.open(im2).convert("RGB")


        # Finding depth : by comparing them all
        disparities = torch.zeros(im1_orig.size).cuda()
        x, y = im1_orig.size

        im1_origt = Image.fromarray(np.array(im1_orig))
        im2_origt = Image.fromarray(np.array(im2_orig))
        w1, h1 = im1_origt.size; w2, h2 = im2_origt.size

        get_matches = GetMatches(model, transform)
        kps1, kps2, prob = get_matches.iterate(im1_origt, im2_origt, iters=2)

        # Then now divide
        kps1 = kps1 / torch.Tensor([w1, h1]).cuda()
        kps2 = kps2 / torch.Tensor([w2, h2]).cuda()

        np.savez_compressed(base_path + '/keypts/%s' % img[:-4], 
                kps1=kps1.cpu().numpy(), 
                kps2=kps2.cpu().numpy(),
                prob=prob.cpu().numpy())

if __name__ == '__main__':
    # Download text files
    download_text()
    # Download images
    download()
    # Preprocess images
    cut_up_images()
