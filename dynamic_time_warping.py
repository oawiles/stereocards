from PIL import Image
import numpy as np
import argparse
import cv2
import os

import time

import torchvision
import torchvision.transforms as tr

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from math import exp

# from models.networks.architectures import VGG19
import threading

from sklearn.linear_model import RANSACRegressor, LinearRegression

# DTW algorithm : really like dynamic programming on making strings match 
def compute_costDTW(n, m, _cost, D=None):
    """
    Part that computes the cost volume for the given images.
    
    Inputs:
    s: array 1 (a 1D numpy array)
    t: array 2 (a 1D numpy array)
    _cost: cost function between two values s[i] and t[j]
    _loc_cost: cost function between two locations (i,j) and (m,l)
    
    Want to find the best alignment between s and t.
    
    Outputs:
    DTW: the weight matrix
    paths: the path through the 2D matrix
    """
    DTW = np.zeros((n, m)) + np.inf
    path = - np.ones((n, m, 2), dtype=np.int)
    
    # Initialise the boundary conditions
    DTW[0,0] = _cost(0,0)
    for i in range(1, n):
        # Now fill in these conditions using the cost function
        DTW[i,0] = _cost(i, 0) + DTW[i-1,0]
        path[i,0,:] = np.array([i-1,0])
        
    for j in range(1, m):
        DTW[0,j] = _cost(0,j) + DTW[0,j-1]
        path[0,j,:] = np.array([0,j-1])
    
    for i in range(1, n):
        # print(i)
        if D is None:
            min_v=1; max_v=m
        else:
            min_v = max(1,i-D)
            max_v = min(m,i+D)
        for j in range(min_v, max_v):
            cost = _cost(i, j)
            
            if DTW[i-1,j] < min(DTW[i, j-1], DTW[i-1,j-1]):
                DTW[i,j] = cost + DTW[i-1,j]
                path[i,j,:] = np.array([i-1,j])
                
            elif DTW[i,j-1] < min(DTW[i-1,j], DTW[i-1,j-1]):
                DTW[i,j] = cost + DTW[i,j-1]
                path[i,j,:] = np.array([i,j-1])

            elif DTW[i-1,j-1] < min(DTW[i-1,j], DTW[i,j-1]):
                DTW[i,j] = cost + DTW[i-1,j-1]
                path[i,j,:] = np.array([i-1,j-1])
                
    return DTW, path

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = VGG19(
            requires_grad=False
        )  # Set to false so that this part of the network is frozen
        self.criterion = nn.L1Loss()
        self.weights = [1.0 , 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def pre_loadfeats(self, im):
        fs = self.model(im)

        return fs

    def forward(self, pred_fs, gt_fs):
        raise Exception("Not implemented!")

        # Collect the losses at multiple layers (need unsqueeze in
        # order to concatenate these together)
        loss = 0
        for i in range(0, len(gt_fs)):
            loss += self.weights[i] * self.criterion(pred_fs[i], gt_fs[i])

        return loss

def perc_sim_features(im1, im2):
    percep_model = PerceptualLoss().cuda()

    # Transform images into appropriate format
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        normalize
    ])

    im1 = transforms(im1).unsqueeze(0).cuda()
    im2 = transforms(im2).unsqueeze(0).cuda()

    with torch.no_grad():
        im1_f = percep_model.pre_loadfeats(im1)
        im2_f = percep_model.pre_loadfeats(im2)

    return im1_f, im2_f, percep_model

# Cost function based on perceptual similarity
def create_percsim(im1, im2, im1_f, im2_f, step, W, model):
    ########################################
    ###### RUNNING PERCEPTUAL SIMILARITY
    ########################################

    def cost_percsim(i, j):
        k = W // 2
        i = i * step
        j = j * step

        loss = 0
        for level_i in range(0, 1):
            _, W1, _ = im1.shape
            _, W2, _ = im2.shape

            _, _, _, WF1 = im1_f[level_i].shape
            _, _, _, WF2 = im2_f[level_i].shape
            
            i_f = int(i * WF1 / float(W1)); j_f = int(j * WF2 / float(W2))
            k_f = int(k * (WF1 + WF2) / (W1 + W2))

            W1_f = im1_f[level_i][:,:,:,max(0,i_f-k_f):i_f+k_f+1]
            W2_f = im2_f[level_i][:,:,:,max(0,j_f-k_f):j_f+k_f+1]
            
            # Pad them to make the same size (pad to the right and down)
            W1_f = F.pad(W1_f, (0, max(0, W2_f.shape[-1] - W1_f.shape[-1]), 
                            0, max(0, W2_f.shape[-2] - W1_f.shape[-2])), mode='constant', value=0)
            W2_f = F.pad(W2_f, (0, max(0, W1_f.shape[-1] - W2_f.shape[-1]),
                            0, max(0, W1_f.shape[-2] - W2_f.shape[-2])), mode='constant', value=0)

            loss += model.criterion(W1_f, W2_f)            

        return loss

    return cost_percsim

# Cost function based on SSIM with a certain window
def create_ssim(im1, im2, step, W):

    ################################
    ###### FIRST CREATE THE WINDOWS
    ################################

    # First create the gaussian window
    def gaussian(window_size, sigma):
        gauss = torch.Tensor(
            [
                exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2))
                    for x in range(window_size)
            ]
        )

        return gauss / gauss.sum()

    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()

        return window

    B = im1.shape[2]
    window = create_window(window_size=11, channel=B)

    def _ssim(img1, img2, window, window_size, channel):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)

        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel)
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel)
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
            - mu1_mu2
        )

        C1 = (0.01) ** 2
        C2 = (0.03) ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        return ssim_map.mean()

    def cost_ssim(i, j):
        k = W // 2
        i = i * step
        j = j * step

        # Look in a region of WxW centered on (i, j)
        W1 = im1[:,max(0,i-k):i+k+1]
        W2 = im2[:,max(0,j-k):j+k+1]

        # Now compare based on SSIM
        W1 = torch.Tensor(W1).unsqueeze(0).permute(0,3,1,2)
        W2 = torch.Tensor(W2).unsqueeze(0).permute(0,3,1,2)
        
        # Pad them to make the same size (pad to the right and down)
        W1 = F.pad(W1, (0, max(0, W2.shape[-1] - W1.shape[-1]), 
                        0, max(0, W2.shape[-2] - W1.shape[-2])), mode='constant', value=0)
        W2 = F.pad(W2, (0, max(0, W1.shape[-1] - W2.shape[-1]),
                        0, max(0, W1.shape[-2] - W2.shape[-2])), mode='constant', value=0)

        # And then compute the SSIM and return
        ssim_W1W2 = _ssim(W1, W2, window, window_size=11, channel=B)

        return 1 - ssim_W1W2

    return cost_ssim

# Cost function based on RGB with a certain window
def create_rgb(im1, im2, step, W, comparison='rgb', normalise=True):
    laplac1 = cv2.Sobel(cv2.cvtColor(np.array(im1), cv2.COLOR_RGB2GRAY),cv2.CV_64F,1,0,ksize=5)
    laplac2 = cv2.Sobel(cv2.cvtColor(np.array(im2), cv2.COLOR_RGB2GRAY),cv2.CV_64F,1,0,ksize=5)

    def cost_rgb(i, j):
        k = W // 2
        i = i * step
        j = j * step
        
        # Look in a region of WxW centered on (i, j)
        W1 = im1[:,max(0, i-k):i+k+1]
        W1 = W1.reshape(-1,3)
        W2 = im2[:,max(0, j-k):j+k+1]
        W2 = W2.reshape(-1,3)
        
        # First do an affine transform to compare regions
        if normalise:
            W1 = (W1 - W1.mean(axis=0)) / np.maximum(W1.std(axis=0), 0.00001)
            W2 = (W2 - W2.mean(axis=0)) / np.maximum(W2.std(axis=0), 0.00001)
        
        # Then compute the L2 err
        N = W1.shape[0]
        M = W2.shape[0]
        err = ((W1[:min(N,M),:] - W2[:min(N,M),:]) ** 2).sum() ** 0.5
        
        return err

    def cost_grad(i, j):
        k = W // 2
        i = i * step
        j = j * step
        
        # Look in a region of WxW centered on (i, j)
        W1 = laplac1[:,max(0, i-k):i+k+1]
        W1 = W1.reshape(-1,)
        W2 = laplac2[:,max(0, j-k):j+k+1]
        W2 = W2.reshape(-1,)
        
        # Then compute the L2 err
        N = W1.shape[0]
        M = W2.shape[0]
        err = ((W1[:min(N,M)] - W2[:min(N,M)]) ** 2).sum() ** 0.5
        
        return err

    def cost_rgbgrad(i, j):
        return cost_rgb(i, j) + cost_grad(i, j)
    
    if comparison == 'rgb':
        return cost_rgb
    elif comparison == 'grad':
        return cost_grad
    elif comparison == 'rgbgrad':
        return cost_rgbgrad

def run_multiprocess_algorithm(im1, im2, step, W, D=None, normalise=True):
    threads = list()

    # Run through each of the y axes:
    N = im1_arr.shape[1]; M = im2_arr.shape[1]
    Y_LEN = im1_arr.shape[0] // step
    sampler = np.zeros((Y_LEN, M//step, 2))

    for y in tqdm(range(0, Y_LEN)):
        x = threading.Thread(target=run_algorithm_singleY, 
                             args=(im1_arr, im2_arr, W, y, step, args, sampler, M, N, D, Y_LEN))
        threads.append(x)
        x.start()

        if len(threads) >= 5:
            for thread in tqdm(threads):
                thread.join()

            threads = []

    return sampler

def run_algorithm_singleY(im1_arr, im2_arr, W, y, step, args, sampler, M, N, D, Y_LEN):
    k = W // 2

    y1_step = y * step

    # Visualise two lines
    line1 = im1_arr[max(0, y1_step-k):y1_step+k+1]
    line2 = im2_arr[max(0, y1_step-k):y1_step+k+1]

    if args.comparison == 'rgb':
        if normalise:
            _cost = create_rgb(line1, line2, step, W, comparison=args.comparison)
        else:
            _cost = create_rgb(line1, line2, step, W, comparison=args.comparison, normalise=False)
    elif args.comparison == 'ssim':
        _cost = create_ssim(line1, line2, step, W)
    elif args.comparison == 'percsim':
        im2F_line = []; im1F_line = []
        for i in range(0, len(im1_F)):
            y1_f = int(y1_step * float(im1_F[i].size(2)) / float(im1_arr.shape[0]))
            k_f = int(k * float(im1_F[i].size(2)) / float(im1_arr.shape[0]))
            im1F_line += [im1_F[i][:,:,max(0,y1_f-k_f):y1_f+k_f+1]]
            im2F_line += [im2_F[i][:,:,max(0,y1_f-k_f):y1_f+k_f+1]]

        _cost = create_percsim(line1, line2, im1F_line, im2F_line, step, W, model)
    elif args.comparison == 'grad':
        _cost = create_rgb(line1, line2, step, W, comparison=args.comparison)
    elif args.comparison == 'rgbgrad':
        _cost = create_rgb(line1, line2, step, W, comparison=args.comparison)
    else:
        raise Exception("Comparison not implemented.... choose from 'rgb', 'ssim' ")

    err, path = compute_costDTW(N // step, M // step, _cost, D=D)

    # And use the path to rewarp one of the images and compare
    t_sampler = np.zeros([err.shape[1], ])

    k = path.shape[0] - 1
    l = path.shape[1] - 1
    for i in range(0, err.shape[0]):
        t_sampler[l] = k

        k, l = path[k,l][0], path[k,l][1]

    xs = (t_sampler * step) / float(line1.shape[1]) * 2 - 1
    sampler[y,:,0] = xs
    sampler[y,:,1] = y / Y_LEN * 2 - 1
    return 

def run_algorithm(im1, im2, step, W, D=None, normalise=True):
    if args.comparison == 'percsim':
        im1_F, im2_F, model = perc_sim_features(im1, im2)

    N = im1_arr.shape[1] 
    M = im2_arr.shape[1] 

    Y_LEN = im1_arr.shape[0] // step
    sampler = np.zeros((Y_LEN,M//step,2))

    for y in tqdm(range(0, Y_LEN)):
        k = W // 2

        y1_step = y * step

        # Visualise two lines
        line1 = im1_arr[max(0, y1_step-k):y1_step+k+1]
        line2 = im2_arr[max(0, y1_step-k):y1_step+k+1]

        if args.comparison == 'rgb':
            if normalise:
                _cost = create_rgb(line1, line2, step, W, comparison=args.comparison)
            else:
                _cost = create_rgb(line1, line2, step, W, comparison=args.comparison, normalise=False)
        elif args.comparison == 'ssim':
            _cost = create_ssim(line1, line2, step, W)
        elif args.comparison == 'percsim':
            im2F_line = []; im1F_line = []
            for i in range(0, len(im1_F)):
                y1_f = int(y1_step * float(im1_F[i].size(2)) / float(im1_arr.shape[0]))
                k_f = int(k * float(im1_F[i].size(2)) / float(im1_arr.shape[0]))
                im1F_line += [im1_F[i][:,:,max(0,y1_f-k_f):y1_f+k_f+1]]
                im2F_line += [im2_F[i][:,:,max(0,y1_f-k_f):y1_f+k_f+1]]

            _cost = create_percsim(line1, line2, im1F_line, im2F_line, step, W, model)
        elif args.comparison == 'grad':
            _cost = create_rgb(line1, line2, step, W, comparison=args.comparison)
        elif args.comparison == 'rgbgrad':
            _cost = create_rgb(line1, line2, step, W, comparison=args.comparison)
        else:
            raise Exception("Comparison not implemented.... choose from 'rgb', 'ssim' ")

        err, path = compute_costDTW(N // step, M // step, _cost, D=D)

        # And use the path to rewarp one of the images and compare
        t_sampler = np.zeros([err.shape[1], ])

        k = path.shape[0] - 1
        l = path.shape[1] - 1
        for i in range(0, err.shape[0]):
            t_sampler[l] = k

            k, l = path[k,l][0], path[k,l][1]

        xs = (t_sampler * step) / float(line1.shape[1]) * 2 - 1
        sampler[y,:,0] = xs
        sampler[y,:,1] = y / Y_LEN * 2 - 1
            
    return sampler

class HorizontalRegression(object):
    def __init__(self, coeffs=None, x=None, y=None, z=None):
        self.coeffs = coeffs
        self.x = x
        self.y = y
        self.z = z

    def fit(self, X, y):
        ys = y.ravel()
        Xs = np.hstack([X, np.ones((X.shape[0], 1))])

        coeffs = np.linalg.lstsq(Xs, ys, rcond=None)[0]

        H = np.array([
            [coeffs[1], - coeffs[0], 0], 
            [coeffs[0], coeffs[1], coeffs[2]], 
            [0, 0, 1] 
        ])

        self.x = coeffs[0]
        self.y = coeffs[1]
        self.z = coeffs[2]

        self.coeffs = H

    def get_params(self, deep=False):
        return {'coeffs': self.coeffs, 'x' : self.x, 'y' : self.y, 'z' : self.z}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def score(self, X, y):
        Xs = np.hstack([X, np.ones((X.shape[0], 1))])
        y_pred = (Xs @ self.coeffs.T)[:,1]

        err = np.abs(y_pred - y).mean()

        return err

    def predict(self, X):
        Xs = np.hstack([X, np.ones((X.shape[0], 1))])
        return (Xs @ self.coeffs.T)[:,1]

    def run(self, X):
        Xs = np.hstack([X, np.ones((X.shape[0], 1))])
        return Xs @ self.coeffs.T

def normalise_xaxis(im1_arr, im2_arr):
    # Find SIFT keypoints and then check if they are truly horizontal
    gray1 = cv2.cvtColor(im1_arr,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2_arr,cv2.COLOR_BGR2GRAY)
    
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(gray1,None)
    kp2, des2 = sift.detectAndCompute(gray2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>10:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    # Use RANSAC in order to allow for a rotation in order to make sure that correspondences
    # lie along horizontal lines
    regressor = HorizontalRegression()
    ransac = RANSACRegressor(regressor, min_samples=3)
    
    # Normalise points
    K1 = np.eye(3)
    K1[0,0] = 2 / im1_arr.shape[1]; K1[1,1] = 2 / im1_arr.shape[0] 
    K1[0:2,2] = - 1 
    K2 = np.eye(3)
    K2[0,0] = 2 / im2_arr.shape[1]; K2[1,1] = 2 / im2_arr.shape[0] 
    K2[0:2,2] = - 1 
    src_pts = K1 @ np.vstack([src_pts[:,0,:].T, np.ones((1,src_pts.shape[0]))])
    dst_pts = K2 @ np.vstack([dst_pts[:,0,:].T, np.ones((1,dst_pts.shape[0]))])

    src_pts = src_pts[0:2].T
    dst_pts = dst_pts[0:2].T

    ransac.fit(src_pts, dst_pts[:,1])

    coeffs = ransac.estimator_.get_params()
    # Undo normalisation
    aff_mat = (np.linalg.inv(K2) @ coeffs['coeffs'] @ K1)

    # Now transform the image and return
    warp_im1 = cv2.warpAffine(im1_arr, aff_mat[0:2], (im2_arr.shape[1], im2_arr.shape[0]))

    return warp_im1, im2_arr, coeffs['coeffs']
    

if __name__ == '__main__':
    arguments = argparse.ArgumentParser()
    arguments.add_argument('--comparison', type=str, default='rgb', help="One of rgb or ssim")
    arguments.add_argument('--im_name', type=str, default='000010')
    arguments.add_argument('--normalise', action='store_true', 
                help="Whether to first use SIFT correspondences in order to ensure correspondences lie on horizontal lines")

    args = arguments.parse_args()

    im_names = sorted(os.listdir(os.environ['BASE_PATH'] + '/imL/'))

    proc_id = int(os.environ['SLURM_NTASKS']) * int(os.environ['SLURM_ARRAY_TASK_ID']) + int(os.environ['SLURM_PROCID'])
    args.im_name = im_names[proc_id][:-4]

    im1 = Image.open(os.environ['BASE_PATH'] + '/imL/%s.jpg' % args.im_name)
    im2 = Image.open(os.environ['BASE_PATH'] + '/imR/%s.jpg' % args.im_name)

    if not os.path.exists(os.environ['BASE_PATH'] + '/warps/'):
        os.makedirs(os.environ['BASE_PATH'] + '/warps/')


    for step in [2]:
        print("Working on step %d ..." % step)
        # Now test it

        ######### FIRST RUN AT A COARSE RESOLUTION
        W = 20
        im1_arr = np.array(im1)
        im2_arr = np.array(im2)

        if args.normalise:
            if not os.path.exists('./cv_temp/'):
                os.makedirs('./cv_temp/')
            cv2.imwrite('./cv_temp/im1.png', cv2.cvtColor(im1_arr, cv2.COLOR_BGR2RGB))
            cv2.imwrite('./cv_temp/im2.png', cv2.cvtColor(im2_arr, cv2.COLOR_BGR2RGB))
            im1_arr, im2_arr, H = normalise_xaxis(im1_arr, im2_arr)
            cv2.imwrite('./cv_temp/im1warp.png', cv2.cvtColor(im1_arr, cv2.COLOR_BGR2RGB))

        a = time.time()
        sampler = run_algorithm(im1_arr, im2_arr, step=step, W=W) 
        b = time.time()
        print(b - a)
        
        # Takes 1 hour to run on the image using the CPU

        if args.normalise:
            suffix = 'norm'
            np.savez_compressed(os.environ['BASE_PATH'] + '/warps/temp_sampler%s_%d_%s_%s.npz' % (
                args.im_name, step, args.comparison, suffix), sampler=sampler, H=H)
        else:
            suffix = 'coarse'


            np.save(os.environ['BASE_PATH'] + '/warps/temp_sampler%s_%d_%s_%s.npz' % (
                args.im_name, step, args.comparison, suffix), sampler)
