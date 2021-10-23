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
import sys

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
def nlm_glcm_filter(im_in, window_radius, patch_radius, h, distances, angles, levels=256, symmetric=False, prop='contrast', max_ram=4):

    """ 
        prop : {'contrast', 'dissimilarity', 'homogeneity', 'energy', \
            'correlation', 'ASM'}, optional 
            
    """
    
    m = im_in.shape[0]
    n = im_in.shape[1]
    
    output = np.empty( (m,n), dtype=np.uint8 )

    im_pad = np.pad( im_in,(patch_radius, patch_radius), mode='symmetric' ).astype(np.float64)
    m_pad = im_pad.shape[0]
    n_pad = im_pad.shape[1]

    kernel = ut.make_kernel(patch_radius)
    eps = 10e-7

    # calculate RAM allocation (in GB)
    total_space = m_pad*n_pad*(levels**2)*distances.shape[0]*angles.shape[0]*8 / 1024**3
    num_iter = int( np.ceil( total_space / max_ram ) )
    X, Y = calc_slices_division( num_iter, m_pad, n_pad)
    # slices dimensions (last is smaller or equal)
    a = m // X
    b = n // Y

    slices = np.empty( (X, Y, a, b ), dtype=np.uint8 )
    slices_pad = np.empty( 
        (X, Y, a+2*patch_radius, b+2*patch_radius), dtype=np.uint8
    )
    slices_out = np.empty_like(slices)

    ut.image2slices(im_in, im_pad, X, Y, slices, slices_pad)

    h = h*h

    I_prop, J_prop = np.ogrid[0:levels, 0:levels]
    I_prop = I_prop.astype(np.float64)
    J_prop = J_prop.astype(np.float64)

    print( f"total RAM needed: {total_space:#.01f} GB. Using maximum of {max_ram:#.01f} GB in {num_iter} iteractions...")

    for ii in range(X):
        for jj in range(Y):

            cur_slice = slices[ ii, jj, :, : ]
            cur_slice_pad = slices_pad[ ii, jj, :, : ]

            # m,n for slices, respectively
            a, b = cur_slice.shape

            im_patch = ut.image2patch(cur_slice, cur_slice_pad, patch_radius)

            glcm_patch, d_patch = patch2glcm(im_patch, a, b, levels, distances, angles, symmetric, prop)

            print( f'processing ({ii+1}, {jj+1}) of ({X}, {Y})... (using {sys.getsizeof(glcm_patch)/1024**3:#.02f} GB for glcm_patch)', end='\t')

            t0 = time.time()
            
            slices_out[ ii, jj, :, : ] = process( cur_slice, cur_slice_pad, 
                glcm_patch, d_patch, kernel, window_radius, patch_radius,
                h, a, b, eps, distances, angles, levels, I_prop, J_prop
            )

            dif = time.time() - t0

            print( f'done in {int(dif//60)} min {dif%60:#.02f} s!')

    output = ut.slices2image(im_in, slices_out)

    return output

@njit(nogil=True, parallel=True)
def process( input, im_pad, glcm_patch, d_patch, kernel, window_radius, patch_radius,
    h, y, x, eps, distances, angles, levels, I_prop, J_prop ):
    
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
                    
                    if( r == offset_y and s == offset_x ):
                        continue
                    if( r >= glcm_patch.shape[0] or s >= glcm_patch.shape[1] ):
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

def patch2glcm(im_patch, m, n, levels, distances, angles, symmetric=False, prop='contrast' ):

    # m and n are obtained from the original (not padded ) image shape
    
    glcm_patch = np.empty((m,n,levels,levels, distances.shape[0], angles.shape[0]), np.uint8)
    d_patch = np.empty( (m, n, distances.shape[0], angles.shape[0]), np.float64 )
    
    for ii in range(m):
        for jj in range(n):
            patch = im_patch[ ii, jj, :, : ]
            glcm = sft.greycomatrix(patch, distances, angles, levels, symmetric)
            glcm_patch[ii, jj, :, :, :, :] = glcm
            d_patch[ ii, jj, :, :] = sft.greycoprops( glcm, prop )

    return glcm_patch, d_patch

def calc_slices_division( N, m, n):

    # system:
    #   (1) N = X*Y
    #   (2) X/Y = m/n 
    # Results in:
    #   Y = sqrt( N * n/m )
    #   X = Y*m/n

    Y = int( np.sqrt( ( n / m ) * N ) )
    X = int( np.ceil( N / Y ) )

    return X, Y