"""
    Credits: Gregory Petterson Zanelato <gregory.zanelato@usp.br>
"""

import numpy as np
import math
import numba
from numba import jit,njit, float64
from utils import *
from scipy.special import expit

from skimage import io
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def nonlocal_means_original(input, window_radius, patch_radius, h):
    m,n=input.shape
    output=np.zeros((m,n), dtype = np.uint8)
    input2 = np.array(np.pad(input,(patch_radius,patch_radius), mode='symmetric'), dtype = np.float64)
    kernel = make_kernel(patch_radius)
    h=h*h
    output = original(input,input2,kernel,window_radius,patch_radius,h,m,n)
    return output


def nonlocal_means_pixelwise(input, h):
    #Adjusting window e p atchsize 
    if(h > 0 and h <= 15):
        patch_radius = 1
        window_radius = 10
        sigma = 0.4*h
    elif ( h > 15 and h <= 30):
        patch_radius = 2
        window_radius = 10
        sigma = 0.4*h
    elif ( h > 30 and h <= 45):
        patch_radius = 3
        window_radius = 17
        sigma = 0.35*h
    elif ( h > 45 and h <= 75):
        patch_radius = 4
        window_radius = 17
        sigma = 0.35*h
    elif (h <= 100):
        patch_radius = 5
        window_radius = 17
        sigma = 0.30*h
    m,n=input.shape
    output=np.zeros((m,n), dtype = np.uint8)
    input2 = np.array(np.pad(input,(patch_radius,patch_radius), mode='symmetric'), dtype = np.float64)
    kernel = make_kernel(patch_radius)
    kernel = kernel / np.sum(kernel)
    h=h*h
    output = pixelwise(input,input2,kernel,window_radius,patch_radius,h,m,n,sigma)
    return output    

@njit()
def original(input, input2, kernel, window_radius, patch_radius, h, y, x):
    output=np.zeros((y,x), dtype = np.uint8)
    patch_size = (2*patch_radius)+1
    w1 = np.zeros((patch_size,patch_size), dtype = np.float64)
    w2 = np.zeros((patch_size,patch_size), dtype = np.float64)
    for i in range(y):
        for j in range(x):
            offset_y = i + patch_radius
            offset_x = j + patch_radius
            
            w1 = input2[offset_y-patch_radius:offset_y+patch_radius+1,offset_x-patch_radius:offset_x+patch_radius+1]
            y_min = np.maximum(offset_y-window_radius,patch_radius)
            y_max = np.minimum(offset_y+window_radius+1,y+patch_radius)
            x_min = np.maximum(offset_x-window_radius,patch_radius)
            x_max = np.minimum(offset_x+window_radius+1,x+patch_radius) 
            wmax = 0
            average  = 0
            sweight = 0
            index_element = 0
            patch_samples_size = ((y_max-y_min)*(x_max-x_min))-1
            weights=np.zeros((patch_samples_size), dtype = np.float64)
            center_patches=np.zeros((patch_samples_size), dtype = np.float64)
            for r in range(y_min,y_max):
                for s in range(x_min,x_max):
                    if(r==offset_y and s==offset_x):
                        continue
                    w2 = input2[r-patch_radius:r+patch_radius+1 , s-patch_radius:s+patch_radius+1]
                    diff = np.subtract(w1,w2)
                    d = calc_distance(kernel,diff)
                    w = calc_weight(d,h)        
                    weights[index_element] = w 
                    center_patches[index_element] = input2[r,s]
                    index_element = index_element + 1 
            wmax = np.max(weights)
            average = np.sum(weights*center_patches) + wmax*input2[offset_y,offset_x]
            sweight = np.sum(weights) + wmax
            if (sweight > 0):                
                output[i,j] = average / sweight
            else:
                output[i,j] = input[i,j]
    return output


@njit()
def pixelwise(input, input2, kernel, window_radius, patch_radius, h, y, x, sigma):
    output=np.zeros((y,x), dtype = np.uint8)
    patch_size = (2*patch_radius)+1
    w1 = np.zeros((patch_size,patch_size), dtype = np.float64)
    w2 = np.zeros((patch_size,patch_size), dtype = np.float64)
    idx_max = 0
    for i in range(y):
        for j in range(x):
            offset_y = i + patch_radius
            offset_x = j + patch_radius
            
            w1 = input2[offset_y-patch_radius:offset_y+patch_radius+1,offset_x-patch_radius:offset_x+patch_radius+1]
            y_min = np.maximum(offset_y-window_radius,patch_radius)
            y_max = np.minimum(offset_y+window_radius+1,y+patch_radius)
            x_min = np.maximum(offset_x-window_radius,patch_radius)
            x_max = np.minimum(offset_x+window_radius+1,x+patch_radius) 
            wmax = 0
            average  = 0
            sweight = 0
            index_element = 0
            patch_samples_size = ((y_max-y_min)*(x_max-x_min))-1
            weights=np.zeros((patch_samples_size), dtype = np.float64)
            center_patches=np.zeros((patch_samples_size), dtype = np.float64)
            for r in range(y_min,y_max):
                for s in range(x_min,x_max):
                    if(r==offset_y and s==offset_x):
                        continue
                    w2 = input2[r-patch_radius:r+patch_radius+1 , s-patch_radius:s+patch_radius+1]
                    dist = np.subtract(w1,w2)
                    d = np.mean((dist)**2) #Euclidean distance
                    max_d = np.maximum(d - 2*(sigma*sigma), 0)
                    w = calc_weight(max_d,h)        
                    weights[index_element] = w 
                    center_patches[index_element] = input2[r,s]
                    index_element = index_element + 1 
            wmax = np.max(weights)
            average = np.sum(weights*center_patches) + wmax*input2[offset_y,offset_x]
            sweight = np.sum(weights) + wmax
            if (sweight > 0):                
                output[i,j] = average / sweight
            else:
                output[i,j] = input[i,j]
    return output