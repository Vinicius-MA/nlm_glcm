"""
    Credits:
        Vinícius Moraes Andreghetti <vinicius.andreghetti@usp.br>
    
    Thanks to:
        Gregory Petterson Zanelato <gregory.zanelato@usp.br>
"""

import numpy as np
from numba import njit, prange, uint64
from skimage import io
from utils import *

""" 

Improving Non-local Means Image Denoising with Gray-Level 
Co-Occurrence Matrix and Haralick Features

    Reference:
        Haralick, RM.; Shanmugam, K., “Textural features for image 
        classification” IEEE Transactions on systems, man, and cybernetics
         6 (1973): 610-621. DOI:10.1109/TSMC.1973.4309314

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

def nlm_glcm_filter(image, window_radius, patch_radius, h, distances, angles, levels, symmetric, normed):
    m,n = image.shape
    output=np.zeros((m,n), dtype = np.uint8)
    input2 = np.array(np.pad(image,(patch_radius,patch_radius), mode='symmetric'), dtype = np.float64)
    kernel = make_kernel(patch_radius)
    eps = 10e-7

    h=h*h

    I_prop, J_prop = np.ogrid[0:levels, 0:levels]
    I_prop = I_prop.astype(np.float64)
    J_prop = J_prop.astype(np.float64)

    output = process(image,input2,kernel,window_radius,patch_radius,h,m,n,eps, distances, angles, levels, symmetric, normed, I_prop, J_prop)
    return output

@njit(nogil=True, parallel=True)
def process(input,input2,kernel,window_radius,patch_radius,h,y,x,eps, distances, angles, levels, symmetric, normed, I_prop, J_prop):
    output=np.zeros((y,x), dtype = np.uint8)
    patch_size = (2*patch_radius)+1
    
    #Initialing patch windows
    w1 = np.zeros((patch_size,patch_size), dtype = np.float64)
    w2 = np.zeros((patch_size,patch_size), dtype = np.float64)
    glcm1 = np.zeros( (levels, levels), dtype=np.uint16 )
    glcm2 = np.zeros( (levels, levels), dtype=np.uint16 )

    done = 0
    
    for i in prange( int(y) ):
        for j in prange( int(x) ):

            offset_y = i + patch_radius
            offset_x = j + patch_radius
            
            #Get NLM central patch in search window (original image)
            w1 = input2[offset_y-patch_radius:offset_y+patch_radius+1,offset_x-patch_radius:offset_x+patch_radius+1]            

            #Calculate boundaries    
            y_min = np.maximum(offset_y-window_radius,patch_radius)             # perguntar para Greg de onde surge esses parâmetros
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
            
            # GLCM and descriptors for central patch
            glcm1 = graycomatrix(w1, distances, angles, levels=levels, symmetric=symmetric, normed=normed)
            d1 = graycoprops(glcm1, I_prop, J_prop, prop="contrast")
            
            #GLCM similarity weight vector - 'w(i,j)GLCM'
            similarity_weights=np.zeros((patch_samples_size), dtype = np.float64)
            
            #Compare central patch with neighbors
            for r in prange( int(y_min), int(y_max) ):
                for s in prange( int(x_min), int(x_max) ):
                    
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
                    
                    # GLCM and descriptors for central patch
                    glcm2 = graycomatrix(w2, distances, angles, levels=levels, symmetric=symmetric, normed=normed)
                    d2 = graycoprops(glcm2, I_prop, J_prop, prop="contrast")
                    
                    #Calculate LBP distance's weight [Kellah - Eq.9]
                    dh = euclidian_distance(d1,d2, eps)
                    similarity_weights[index_element] = dh
                    index_element = index_element + 1 

            #Sampled standard deviation of all LBP similarity distances obtained according to [Kellah - Eq.9].               
            hSi = np.std(similarity_weights) + eps
            #LBP Weighting function - [Kellah - Eq.8]           
            similarity_weights = calc_weight(similarity_weights,h)    
         
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
            
            done += 1
            print(100*done/(y*x) , '% done')
    
    return output
