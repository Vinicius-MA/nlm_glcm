"""
    Credits:
        Vinícius Moraes Andreghetti <vinicius.andreghetti@usp.br>
    
    Thanks to:
        Gregory Petterson Zanelato <gregory.zanelato@usp.br>
"""

import numpy as np
from numba import njit, prange, cuda
import skimage.feature as sft
import utils as ut
import time

"""

Improving Non-local Means im_in Denoising with Gray-Level 
Co-Occurrence Matrix and Haralick Features

    Reference:
        Haralick, RM.; Shanmugam, K., “Textural features for im_in 
        classification” IEEE Transactions on systems, man, and cybernetics
         6 (1973): 610-621. DOI:10.1109/TSMC.1973.4309314

# ------------------------------------------------------------------------
# Inputs                                           
#   im_in          : Noisy image
#   window_radius  : Search window size
#   patch_radius   : Patch distance
#   h              : Filter degree NLM
#   lbp_method     : LBP method 
#   lbp_n_points   : LBP sample points
#   lbp_radius     : LBP Radius  
# ------------------------------------------------------------------------
"""
def nlm_glcm_filter(im_in, window_radius, patch_radius, h, distances, angles, levels, symmetric, normed):

    m = im_in.shape[0]
    n = im_in.shape[1]

    print( 'calculating glcm...', end=' ')
    
    output = np.empty( (m,n), dtype=np.uint8 )

    im_pad = np.pad( im_in,(patch_radius, patch_radius), mode='symmetric' ).astype(np.float64)
    kernel = ut.make_kernel(patch_radius)
    eps = 10e-7
    
    t0 = time.time()
    im_patch = ut.image2patch(im_in, im_pad, patch_radius)
    glcm_patch, d_patch = patch2glcm(im_patch, m, n, levels, distances, angles)
    t1 = time.time()
    print(f'done in {t1-t0:#.03f} s!')

    h = h*h

    I_prop, J_prop = np.ogrid[0:levels, 0:levels]
    I_prop = I_prop.astype(np.float64)
    J_prop = J_prop.astype(np.float64)

    print('processing image...', end=' ')
    t0 = time.time()
    output = process(im_in,im_pad, im_patch, glcm_patch, d_patch, kernel,window_radius,patch_radius,
        h,m,n,eps, distances, angles, levels, symmetric, normed, I_prop, J_prop )
    t1 = time.time()

    print(f'done in {t1-t0:#.03f} s!')


    return output

@njit(nogil=True, parallel=True)
def process( input, im_pad, im_patch, glcm_patch, d_patch, kernel, window_radius, patch_radius,
    h, y, x, eps, distances, angles, levels, symmetric, normed, I_prop, J_prop ):
    
    output=np.empty( (y,x), dtype=np.uint8 )
    patch_size = (2*patch_radius)+1
    
    #Initialing patch windows
    w1 = np.zeros((patch_size,patch_size), dtype = np.float64)
    w2 = np.zeros((patch_size,patch_size), dtype = np.float64)
    glcm1 = np.zeros( (levels, levels), dtype=np.uint16 )
    glcm2 = np.zeros( (levels, levels), dtype=np.uint16 )

    done = 0
    
    for i in prange( y ):
        for j in prange( x ):

            offset_y = i + patch_radius
            offset_x = j + patch_radius
            
            #Get NLM central patch in search window (original im_in)
            w1 = im_pad[offset_y-patch_radius:offset_y+patch_radius+1,offset_x-patch_radius:offset_x+patch_radius+1]

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
            intensity_weights=np.zeros((patch_samples_size), dtype=np.float64)
            center_patches=np.zeros((patch_samples_size), dtype=np.float64)
            
            # GLCM and descriptors for central patch
            glcm1 = glcm_patch[i, j, :, :, :, :]
            d1 = d_patch[i, j, :, :]
            
            #GLCM similarity weight vector - 'w(i,j)GLCM'
            similarity_weights=np.zeros((patch_samples_size), dtype=np.float64)
            
            #Compare central patch with neighbors
            for r in prange( int(y_min), int(y_max) ):
                for s in prange( int(x_min), int(x_max) ):
                    
                    if(r==offset_y and s==offset_x):
                        continue
                    
                    #Get NLM neighbor patch in search window (original im_in)
                    w2 = im_pad[r-patch_radius:r+patch_radius+1 , s-patch_radius:s+patch_radius+1]
                    
                    #Calculate NLM distance's weight
                    diff = np.subtract(w1,w2)
                    d = ut.calc_distance(kernel,diff)
                    w = ut.calc_weight(d,h)        
                    intensity_weights[index_element] = w 
                    center_patches[index_element] = im_pad[r,s]
                    
                    # GLCM and descriptors for central patch
                    glcm2 = glcm_patch[r, s, :, :, :, : ]
                    d2 = d_patch[r, s, :, :]
                    
                    #Calculate LBP distance's weight [Kellah - Eq.9]
                    dh = ut.euclidian_distance(d1,d2, eps)
                    similarity_weights[index_element] = dh
                    index_element = index_element + 1 

            #Sampled standard deviation of all LBP similarity distances obtained according to [Kellah - Eq.9].               
            hSi = np.std(similarity_weights) + eps
            #LBP Weighting function - [Kellah - Eq.8]           
            similarity_weights = ut.calc_weight(similarity_weights,h)    
         
            #NLM max central pixel
            wmax = np.max(intensity_weights)
            
            #Modulated weights - [Kellah - Eq.3]
            modulated_weights = intensity_weights*similarity_weights
            average = np.sum(modulated_weights*center_patches) + wmax*im_pad[offset_y,offset_x]
            sweight = np.sum(modulated_weights) + wmax
            if (sweight > 0):                
                output[i,j] = average / sweight
            else:
                output[i,j] = input[i,j]
    
    return output

def patch2glcm(im_patch, m, n, levels, distances, angles, prop='contrast' ):

    # m and n are obtained from the original (not padded ) image shape
    
    glcm_patch = np.empty((m,n,levels,levels, distances.shape[0], angles.shape[0]), np.uint8)
    d_patch = np.empty( (m, n, distances.shape[0], angles.shape[0]), np.float64 )
    
    for ii in range(m):
        for jj in range(n):
            patch = im_patch[ ii, jj, :, : ]
            glcm = sft.greycomatrix(patch, distances, angles, levels)
            glcm_patch[ii, jj, :, :, :, :] = glcm
            d_patch[ ii, jj, :, :] = sft.greycoprops( glcm, prop )

    return glcm_patch, d_patch