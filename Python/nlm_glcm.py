""" Non-Local Means with GLCM wighting improvement.
    Credits:
     Vinícius Moraes Andreghetti <vinicius.andreghetti@usp.br>
    
    Special thanks to: 
     Gregory Petterson Zanelato <gregory.zanelato@usp.br>"""

import sys
import time

import numpy as np
import skimage.feature as sft
from numba import cuda, njit, prange

import utils as ut


def nlm_glcm_filter(im_in, window_radius, patch_radius, sigma, 
     distances, angles, levels=256, props=['contrast'],
     symmetric=False, max_ram_gb=4.
    ) :
    """Performs the Non-Local Means with Gray-Level Co-Occurrence 
     Matrix (GLCM) weighting improvement, on 2D grayscale images.

     Parameters:
     ---------------     
     * im_in: 2D ndarray [np.uint8]
        Input noisy image
     * window_radius: int
        Radius of the searching windows (Si) at which the pixels will
         be considered to calculate the central pixel's output value.    
     * patch_radius: int
        Radius of the patches (Li, Lj) that will be compared to 
         compute distances.
     * sigma: float
        Estimative of the parameter sigma of the im_in's noise 
         distribution.    
     * distances: 1D ndarray [np.uint8]
        List of pixel pair distance offsets, in pixels, to be 
         considered at the GLCM calculus.    
     * angles: 1D ndarray [np.float64]
        List of pixel pair angles, in radians, to be considered at 
         the GLCM calculus.    
     * levels: int, optional
        The input image should contain integers in [0, `levels`-1],
         where levels indicate the number of gray-levels counted
         (typically 256 for an 8-bit image). This argument is 
         required for 16-bit images or higher and is typically the 
         maximum of the image. As the output matrix is at least 
         `levels` x `levels`, it might be preferable to use binning
         of the input image rather than large values for `levels`.    
     * props: array_like [str], optional
        List of strings that indicates which GLCM properties will be
         computed at the filter. The possible values are:
            ['contrast', 'dissimilarity', 'homogeneity', 'energy',
            'correlation', 'ASM' ]
         Default value is: ['contrast']
     * symmetric: bool, optional
        If True, the GLCM matrix `G[:, :, d, theta]` is symmetric. 
         This is accomplished by ignoring the order of value pairs, 
         so both (i, j) and (j, i) are accumulated when (i, j) is 
         encountered for a given offset. The default is False.    
     * max_ram_gb: float, optional (is it correctly implemented?)
        Maximum amount of RAM memory to be available to store the
         glcm_patch array the (probably unnecessarily) biggest array
         in this algorithm. Accordingly to this value, the input
         image is broken into minor slices, that are processed 
         independently.
    
     Returns:
     ---------------
     * output: 2D ndarray [np.uint8]
        The final processed calculated image.
    
     References:
     ---------------
     [1] A. Buades, B. Coll and J. -. Morel, "A non-local algorithm
      for image denoising," 2005 IEEE Computer Society Conference on
      Computer Vision and Pattern Recognition (CVPR'05), 2005, pp. 
      60-65 vol. 2, doi: 10.1109/CVPR.2005.38.
     [2] Haralick, RM.; Shanmugam, K., “Textural features for image 
      classification” IEEE Transactions on systems, man, and 
      cybernetics 6 (1973): 610-621. DOI:10.1109/TSMC.1973.4309314
     [3] Khellah, Fakhry. (2013). Application of Local Binary Pattern
      to Windowed Nonlocal Means Image Denoising. 8156. 21-30. 10.1007/978-3-642-41181-6_3.
     
     """
    
    # input image shape is (m,n)
    m = im_in.shape[0]
    n = im_in.shape[1]

    # padding input image and get new shape
    im_pad = np.pad( im_in,(patch_radius, patch_radius),
        mode='symmetric' ).astype(np.float64
    )
    m_pad = im_pad.shape[0]
    n_pad = im_pad.shape[1]

    # make kernel
    kernel = ut.make_kernel(patch_radius)
    eps = 10e-7

    # calculate RAM allocation
    total_space_gb = ( m_pad*n_pad*(levels**2)
        *distances.shape[0]*angles.shape[0] / 1024**3
    )
    num_iter = int( np.ceil( total_space_gb / max_ram_gb ) )
    X, Y = calc_slices_division( num_iter, m_pad, n_pad)
    # updates num_iter with calculated X,Y
    num_iter = X*Y    
    # slices dimensions (last slice is smaller or equal)
    a = m // X
    b = n // Y

    # alocate arrays memory
    output = np.empty( (m,n), dtype=np.uint8 )
    slices = np.empty( (X,Y,a,b), dtype=np.uint8 )
    slices_pad = np.empty( 
        (X, Y, a+2*patch_radius, b+2*patch_radius), dtype=np.uint8
    )
    slices_out = np.empty_like(slices)

    # break input image into slices (avoid RAM overflow)
    ut.image2slices(im_in, im_pad, slices, slices_pad)

    # get NLM smoothness parameter
    h = sigma*sigma

    print( f"\ttotal RAM: {total_space_gb:#.01f} GB. Max:{max_ram_gb:#.01f} GB ({num_iter} iteractions)")

    cur_iter = 0
    # process each slice
    for ii in range(X):
        for jj in range(Y):

            if ii != 0 or jj != 0:
                print( f'\t({ii+1}, {jj+1}) of ({X}, {Y})...', end='')

            # current slice and padded slice
            cur_slice = slices[ ii, jj, :, : ]
            cur_slice_pad = slices_pad[ ii, jj, :, : ]

            # cur_slice shape
            a, b = cur_slice.shape

            # get patch array from cur_slice
            t0 = time.time()            
            im_patch = ut.image2patch(cur_slice, cur_slice_pad, patch_radius)
            dif0 = time.time() - t0
            
            # get GLCM array from patch array (critical RAM point)
            t0 = time.time()
            glcm_patch, d_patch = patch2glcm(im_patch, a, b, levels,
                distances, angles, props, symmetric
            )
            dif1 = time.time() - t0

            if ii == 0 and jj == 0:
                print( f'\tallocating {sys.getsizeof(glcm_patch)/1024**3:#.02f} GB for glcm_patch')
                print( f'\t({ii+1}, {jj+1}) of ({X}, {Y})...', end='')

            # process current slice and store it to the slice_out array
            t0 = time.time()            
            slices_out[ ii, jj, :, : ] = process( cur_slice, cur_slice_pad, 
                glcm_patch, d_patch, kernel, window_radius, patch_radius,
                h, a, b, eps, distances, angles, levels
            )
            dif2 = time.time() - t0

            cur_iter += 1

            print( f'\tdone:{dif0:05.02f}s + {int(dif1//60):02d}:{int(dif1%60):02d} + {dif2:05.02f}s')

    # finally, get output image from slices_out array
    output = ut.slices2image(im_in, slices_out)

    return output

@njit(nogil=True, parallel=True)
def process( input, im_pad, glcm_patch, d_patch, kernel, window_radius, 
     patch_radius, h, y, x, eps, distances, angles, levels 
    ):
    
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
            w1 = im_pad[
                offset_y-patch_radius:offset_y+patch_radius+1,
                offset_x-patch_radius:offset_x+patch_radius+1
            ]

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
                    w2 = im_pad[
                        r-patch_radius:r+patch_radius+1,
                        s-patch_radius:s+patch_radius+1
                    ]
                    
                    #Calculate NLM distance's weight
                    diff = np.subtract(w1,w2)
                    d = ut.calc_distance(kernel,diff)
                    w = ut.calc_weight(d,h)        
                    intensity_weights[index_element] = w 
                    center_patches[index_element] = im_pad[r,s]
                    
                    # GLCM and descriptors for central patch
                    glcm2 = glcm_patch[r, s, :, :, :, : ]
                    d2 = d_patch[r, s, :, :]
                    
                    #Calculate GLCM distance's weight [Kellah - Eq.9]
                    dh = ut.euclidian_distance( d1, d2, eps )
                    #dh = ut.chisquare_distance( d1, d2, eps )
                                        
                    similarity_weights[index_element] = dh
                    index_element = index_element + 1 

            #Sampled standard deviation of all LBP similarity distances obtained 
            # according to [Kellah - Eq.9].               
            hSi = np.std(similarity_weights) + eps

            #LBP Weighting function - [Kellah - Eq.8]           
            #similarity_weights = ut.calc_weight( similarity_weights, h )
            similarity_weights = ut.calc_weight( similarity_weights, hSi )
         
            #NLM max central pixel
            wmax = np.max(intensity_weights)
            
            #Modulated weights - [Kellah - Eq.3]
            modulated_weights = intensity_weights * similarity_weights
            average = (
                np.sum( modulated_weights*center_patches) +
                wmax*im_pad[offset_y,offset_x]
            )
            
            sweight = np.sum(modulated_weights) + wmax
            if (sweight > 0):                
                output[i,j] = average / sweight
            else:
                output[i,j] = input[i,j]
    
    return output

def patch2glcm(im_patch, m, n, levels, distances, angles, props,
     symmetric=False
    ):
    """Performs...

     Parameters:
     ---------------
     * im_patch: 4D ndarray [np.uint8]
     * m: int
     * n: int
     * levels: int, optional
        The input image should contain integers in [0, `levels`-1],
         where levels indicate the number of gray-levels counted
         (typically 256 for an 8-bit image). This argument is 
         required for 16-bit images or higher and is typically the 
         maximum of the image. As the output matrix is at least 
         `levels` x `levels`, it might be preferable to use binning
         of the input image rather than large values for `levels`.
     * distances: 1D ndarray [np.uint8]
        List of pixel pair distance offsets, in pixels, to be 
         considered at the GLCM calculus.
     * angles: 1D ndarray [np.float64]
        List of pixel pair angles, in radians, to be considered at 
         the GLCM calculus.
     * props: array_like [str], optional
        List of strings that indicates which GLCM properties will be
         computed at the filter. The possible values are:
            ['contrast', 'dissimilarity', 'homogeneity', 'energy',
            'correlation', 'ASM' ]
     * symmetric: bool, optional
        If True, the GLCM matrix `G[:, :, d, theta]` is symmetric. 
         This is accomplished by ignoring the order of value pairs, 
         so both (i, j) and (j, i) are accumulated when (i, j) is 
         encountered for a given offset. The default is False.
     
    
     Returns:
     ---------------
     * glcm_patch: 6D ndarray [np.uint8]
     * d_patch: 4D ndarray [np.float64]

     References:
     ---------------
     
     """
    
    # m and n are obtained from the original (not padded ) image shape

    glcm_patch = np.empty((m,n,levels,levels, distances.shape[0], angles.shape[0]), np.uint8)
    d_patch = np.empty( (m, n, distances.shape[0], angles.shape[0]), np.float64 )
    
    for ii in range(m):
        for jj in range(n):

            patch = im_patch[ ii, jj, :, : ]
            glcm = sft.greycomatrix(patch, distances, angles, levels, symmetric)
            glcm_patch[ii, jj, :, :, :, :] = glcm

            d = np.ones( ( distances.shape[0], angles.shape[0]), np.float64 )
            for prop in props:
                d *= sft.greycoprops( glcm, prop )
            
            d_patch[ ii, jj, :, :] =  d

    return glcm_patch, d_patch

def calc_slices_division( num_iter, m, n):

    # system:
    #   (1) N = X*Y
    #   (2) X/Y = m/n 
    # Results in:
    #   Y = sqrt( N * n/m )
    #   X = Y*m/n

    Y = int( np.ceil( np.sqrt( ( n / m ) * num_iter ) ) )
    X = int( np.ceil( num_iter / Y ) )

    return X, Y
