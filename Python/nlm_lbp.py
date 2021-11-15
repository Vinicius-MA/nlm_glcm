"""
    Credits: Gregory Petterson Zanelato <gregory.zanelato@usp.br>
"""

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from skimage.feature import local_binary_pattern

from utils import *

""" 
F. Khellah, 'Application of local binary pattern to windowed nonlocal
means image denoising,' Image Analysis and Processing (Iciap 2013),
Pt 1, vol. 8156, pp. 21–30, 2013

# ------------------------------------------------------------------------
# Inputs                                           
#   input          : Noisy Image
#   window_radius  : Search window size
#   patch_radius   : Patch distance
#   h              : Filter degree NLM
#   lbp_method     : LBP method 
#   lbp_n_points   : LBP sample points
#   lbp_radius     : LBP Radius  
# ------------------------------------------------------------------------
"""

def nonlocal_means_lbp_original(input, window_radius, patch_radius, h, lbp_method, lbp_n_points, lbp_radius):
    m,n=input.shape
    output=np.zeros((m,n), dtype = np.uint8)
    input2 = np.array(np.pad(input,(patch_radius,patch_radius), mode='symmetric'), dtype = np.float64)
    kernel = make_kernel(patch_radius)
    eps = 10e-7
    
    img_lbp = local_binary_pattern(image=input2, P=lbp_n_points, R=lbp_radius, method=lbp_method)
    h=h*h
    output = process(input,input2,kernel,window_radius,patch_radius,h,m,n,img_lbp,eps, lbp_n_points)
    return output


@njit()
def process(input, input2, kernel, window_radius, patch_radius, h, y, x, descriptor_img, eps, lbp_n_points):
    output=np.zeros((y,x), dtype = np.uint8)
    patch_size = (2*patch_radius)+1
    
    #Initialing patch windows
    w1 = np.zeros((patch_size,patch_size), dtype = np.float64)
    w2 = np.zeros((patch_size,patch_size), dtype = np.float64)
    w1_descriptor = np.zeros((patch_size,patch_size), dtype = np.float64)
    w2_descriptor = np.zeros((patch_size,patch_size), dtype = np.float64)
    
    print('\tnlm-lbp: total ', y,',',x)

    #Initialing histogram distance vector (LBP)
    n_bins = lbp_n_points+2
    h_li =  np.zeros((n_bins,), dtype = np.float64)
    h_lj =  np.zeros((n_bins,), dtype = np.float64)
    
    for i in range(y):

        #print('\t\tline ',i+1, ' out of ', y)

        for j in range(x):

            offset_y = i + patch_radius
            offset_x = j + patch_radius
            
            #Get NLM central patch in search window (original image)
            w1 = input2[offset_y-patch_radius:offset_y+patch_radius+1,offset_x-patch_radius:offset_x+patch_radius+1]
            
            #Get LBP central patch in search window (LBP Features image)
            w1_descriptor = descriptor_img[offset_y-patch_radius:offset_y+patch_radius+1,offset_x-patch_radius:offset_x+patch_radius+1]
            #Generate Histogram
            h_li = np.histogram(w1_descriptor, bins=n_bins)[0]

            #Calculate boundaries    
            y_min = np.maximum(offset_y-window_radius,patch_radius)
            y_max = np.minimum(offset_y+window_radius+1,y+patch_radius)
            x_min = np.maximum(offset_x-window_radius,patch_radius)
            x_max = np.minimum(offset_x+window_radius+1,x+patch_radius) 
            wmax = 0
            average  = 0
            sweight = 0
            index_element = 0
            patch_samples_size = ((y_max-y_min)*(x_max-x_min))-1
            
            #NLM intensity weight vector  - 'w(i,j)intensity'
            intensity_weights=np.zeros((patch_samples_size), dtype = np.float64)
            center_patches=np.zeros((patch_samples_size), dtype = np.float64)
            
            #LBP similarity weight vector - 'w(i,j)LBP'
            similarity_weights=np.zeros((patch_samples_size), dtype = np.float64)
            
            #Compare central patch with neighbors
            for r in range(y_min,y_max):
                for s in range(x_min,x_max):
                    if(r==offset_y and s==offset_x):
                        continue
                    
                    #Get NLM neighbor patch in search window (original image)
                    w2 = input2[r-patch_radius:r+patch_radius+1 , s-patch_radius:s+patch_radius+1]
                    
                    #Calculate NLM distance's weight
                    diff = np.subtract(w1,w2)
                    d = calc_distance(kernel,diff)
                    w = calc_weight(d,h)        
                    intensity_weights[index_element] = w 
                    center_patches[index_element] = input2[r,s]

                    #Get LBP neighbor patch in search window (LBP Features image)
                    w2_descriptor = descriptor_img[r-patch_radius:r+patch_radius+1 , s-patch_radius:s+patch_radius+1]
                    #Generate Histogram
                    h_lj = np.histogram(w2_descriptor, bins=n_bins)[0]
                    
                    #Calculate LBP distance's weight [Kellah - Eq.9]
                    dh = chisquare_distance(h_li,h_lj, eps)
                    similarity_weights[index_element] = dh
                    index_element = index_element + 1 
            #Sampled standard deviation of all LBP similarity distances obtained according to [Kellah - Eq.9].               
            hsi = np.std(similarity_weights) + eps
            #LBP Weighting function - [Kellah - Eq.8]           
            similarity_weights = calc_weight(similarity_weights,hsi)    
         
            #NLM max central pixel
            wmax = np.max(intensity_weights)
            
            #Modulated weights - [Kellah - Eq.3]
            modulated_weights = intensity_weights*similarity_weights
            average = np.sum(modulated_weights*center_patches) + wmax*input2[offset_y,offset_x]
            sweight = np.sum(modulated_weights) + wmax
            if (sweight > 0):                
                output[i,j] = average / sweight
            else:
                output[i,j] = input[i,j]

    return output
