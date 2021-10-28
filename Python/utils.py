"""
    Credits: Gregory Petterson Zanelato <gregory.zanelato@usp.br>

    References:
        https://scikit-image.org/
"""

import math

import numba
import numpy as np
from numba import float64, jit, njit, prange, uint64
from numba.typed import Dict

@njit()
def normalize_data(data,eps):
    return (data - np.min(data)) / (np.max(data) - np.min(data) + eps)

@njit()
def stein(s,sigma):
    return np.exp(-s/sigma)

@njit()
def chisquare_distance(A, B, eps): 
    return np.sum( (A-B)**2 / (A+B+eps) )

@njit()
def euclidian_distance(A, B, eps):
    return np.sum( np.sqrt( (A - B)**2 ) )

def calculate_psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float64) - np.array(img2, dtype=np.float64)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))

def add_gaussian_noise(image, mean=0, sigma=20, max_gray=255):
    """Add Gaussian noise to an image of type np.uint8."""
    gaussian_noise = np.random.normal(mean, sigma, image.shape)
    gaussian_noise = gaussian_noise.reshape(image.shape)
    noisy_image = image + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, max_gray)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image

@njit()
def calc_distance(kernel,diff):
    return np.sum(kernel*(diff)*(diff))

@njit()
def calc_weight(d,h):
    return np.exp(-d/h)

@njit()
def calc_average(weights, center_patches, img):
    return (np.sum(weights*center_patches) + np.max(weights)*img)
   
@njit()   
def calc_sweight(weights):
    return (np.sum(weights) + np.max(weights))
    
@njit()
def make_kernel(f):              
    kernel=np.zeros((2*f+1,2*f+1), dtype=np.float64)   
    for d in range(1,f+1):
        value = 1 / (2*d+1)**2     
        for i in range(-d,d+1):
            for j in range(-d,d+1):
                kernel[f-i,f-j] = kernel[f-i,f-j] + value 
    result =  (kernel / f)
    return (kernel / f)

@njit(nogil=True)
def graycomatrix(image, distances, angles, levels, symmetric, normed):

    G = np.zeros( (
            np.uint16(levels), np.uint16(levels),
            np.uint16(len(distances)),np.uint16(len(angles)) 
         ), dtype=np.float64
    )

    # count co-occurrences
    glcm_loop(image, distances, angles, levels, G)

    # make G symmetric
    if symmetric:
        Gt = np.transpose( G, (1, 0, 2, 3) )
        G += Gt
    
    # normalize G
    if normed:
        Gn = G.astype( np.float64 )
        #glcm_sums = np.sum( Gn, axis=(0,1) )
        glcm_sums = np.sum( np.sum(Gn, axis=0), axis=0)
        for elem in np.nditer(glcm_sums):
            if elem == 0:
                elem = 1
        G = Gn / glcm_sums

    return G

@njit(nogil=True)
def glcm_loop(image, distances, angles, levels, G):
    rows, cols = image.shape
    for a_idx in range( len(angles) ):
        angle = angles[ a_idx ]
        for d_idx in range( len(distances) ):
            dist = distances[ d_idx ]
            offset_row = round( np.sin(angle) * dist )
            offset_col = round( np.cos(angle) * dist )
            start_row = np.uint16( max( 0, -offset_row ) )
            end_row = np.uint16( min( rows, rows - offset_row ) )
            start_col = np.uint16( max( 0, -offset_col ) )
            end_col = np.uint16( min( cols, cols - offset_col ) )
            for r in prange( start_row, end_row ):
                for c in prange( start_col, end_col ):
                    i = np.uint16( image[r, c] )
                    # computes pixel position
                    j_row = r + offset_row
                    j_col = c + offset_col
                    # comparing pixel
                    j = np.uint16( image[ j_row, j_col ] )
                    if ( i >= 0 and i< levels) and ( j >= 0 and j < levels ):
                        G[i, j, d_idx, a_idx] += 1
    dummy = 0

@njit(nogil=True)
def graycoprops(G, I, J, num_level=256, prop='contrast'):
    
    (num_level1, num_level2, num_dist, num_angle) = G.shape
    #if num_level1 != num_level or num_level2 != num_level:
    #    print("graycoprops: G dimensions must be equal to num_level. Dims: (", num_level1,",", num_level2, "). num_level: ", num_level)

    # normalize G
    Gn = G.astype( np.float64 )
    glcm_sums = np.sum( np.sum(Gn, axis=0), axis=0)
    for elem in np.nditer(glcm_sums):
        if elem == 0:
            elem = 1
    Gn /= glcm_sums

    # create weights for specified properties
    #I, J = np.ogrid[0:num_level, 0:num_level]
    if prop == 'contrast':
        weights = (I - J) ** 2
    elif prop == 'dissimilarity':
        weights = np.abs(I - J)
    elif prop == 'homogeneity':
        weights = 1. / (1. + (I - J) ** 2)
    elif prop in ['ASM', 'energy', 'correlation']:
        pass
    else:
        print("Value Error: ", prop)
    
    # compute property for each GLCM
    if prop == 'energy':
        
        weights = weights.reshape((num_level, num_level, 1, 1))
        results = np.sum( np.sum(Gn * weights, axis=0), axis=0 )
        
        ## bugado com njit
        #asm = np.sum(G ** 2, axis=(0, 1))
        #results = np.sqrt(asm)
    elif prop == 'ASM':
        
        weights = weights.reshape((num_level, num_level, 1, 1))
        results = np.sum( np.sum(Gn * weights, axis=0), axis=0 )

        ### bugado com njit
        #results = np.sum(G ** 2, axis=(0, 1))
    elif prop == 'correlation':
        
        weights = weights.reshape((num_level, num_level, 1, 1))
        results = np.sum( np.sum(Gn * weights, axis=0), axis=0 )

        """
        #results = np.zeros((num_dist, num_angle), dtype=np.float64)            ## buga njit

        aux = np.array( [ x for x in range(num_level) ], dtype=np.float64 )
        I = aux.reshape((num_level, 1, 1, 1) )
        J = aux.reshape((1, num_level, 1, 1) )
        diff_i = I - np.sum(I * G, axis=(0, 1))
        diff_j = J - np.sum(J * G, axis=(0, 1))

        std_i = np.sqrt(np.sum(G * (diff_i) ** 2, axis=(0, 1)))
        std_j = np.sqrt(np.sum(G * (diff_j) ** 2, axis=(0, 1)))
        cov = np.sum(G * (diff_i * diff_j), axis=(0, 1))

        # handle the special case of standard deviations near zero
        mask_0 = ( std_i < 1e-15 ) or ( std_j < 1e-15 )
        mask_0[std_j < 1e-15] = True
        results[mask_0] = 1                ## não funciona com jnit

        # handle the standard case
        mask_1 = ~mask_0
        results[mask_1] = cov[mask_1] / (std_i[mask_1] * std_j[mask_1])            ### não funciona com njit
        """

    elif prop in ['contrast', 'dissimilarity', 'homogeneity']:
        weights = weights.reshape((num_level, num_level, 1, 1))
        results = np.sum( np.sum(Gn * weights, axis=0), axis=0 )

    return results

def gaussian_noise(size, mean=0, std=0.01):
    '''
    Generates a matrix with Gaussian noise in the range [0-255] to be added to an image
    
    :param size: tuple defining the size of the noise matrix 
    :param mean: mean of the Gaussian distribution
    :param std: standard deviation of the Gaussian distribution, default 0.01
    :return matrix with Gaussian noise to be added to image
    '''
    noise = np.multiply(np.random.normal(mean, std, size), 255)
    
    return noise

def gaussian_kernel(k=3, sigma=1.0):
    arx = np.arange((-k//2) + 1.0, (k//2) + 1.0)
    x, y = np.meshgrid(arx, arx)
    filt = np.exp(-(1/2)*(np.square(x) + np.square(y))/np.square(sigma))
    return filt/np.sum(filt)

@njit(nogil=True, parallel=True)
def image2patch( im_in, im_pad, hw ):
    
    m = im_in.shape[0]
    n = im_in.shape[1]
    w = 2*hw + 1

    im_patch = np.empty( (m, n, w, w), dtype=np.uint8 )

    for ii in prange(m):
        for jj in prange(n):
            im_patch[ ii, jj, 0:w, 0:w ] = im_pad[ ii:ii+w, jj:jj+w ].astype(np.uint8)

    return im_patch

"""
parameters:
    im_in : input image (gray-scale - 2D, uint8)
    X :     number of width-oriented slices
    Y :     number of height-oriented slices
    hw:     image padding half-width
 return (to):
    slices :slice-array refering to im_in
    slices_pad: slice-array referring to padded input image (im_pad)
"""
@njit(nogil=True, parallel=True)
def image2slices( im_in, im_pad, slices, slices_pad ):

    im_in = im_in.astype( np.uint8 )
    
    m = im_in.shape[0]
    n = im_in.shape[1]

    # slices dimensions (last is less or equal)
    X = slices.shape[0]
    Y = slices.shape[1]
    a = slices.shape[2]
    b = slices.shape[3]

    w = slices_pad.shape[2] - a

    for ii in prange(X):
        for jj in prange(Y):

            # current slice boundaries
            start_m = ii * a
            end_m = np.minimum( (ii+1)*a, m )
            start_n = jj * b
            end_n = np.minimum( (jj+1)*b, n )

            #  the last slice dimensions can be less than a x b
            diff_m = end_m - start_m
            diff_n = end_n - start_n

            slices[ ii, jj, 0 : diff_m, 0 : diff_n ] = (
                 im_in[ start_m : end_m, start_n : end_n ] 
                )

            slices_pad[ ii, jj, 0:diff_m + w, 0:diff_n + w] = (
                im_pad[ start_m : end_m + w, start_n : end_n + w]
            )

@njit(nogil=True, parallel=True)
def slices2image( im_in, slices ):

    X = slices.shape[0]
    Y = slices.shape[1]
    a = slices.shape[2]
    b = slices.shape[3]

    m = im_in.shape[0]
    n = im_in.shape[1]

    im_out = np.empty_like(im_in)

    for ii in prange(X):
        for jj in prange(Y):

            # current slice boundaries
            start_m = ii * a
            end_m = np.minimum( (ii+1)*a, m )
            start_n = jj * b
            end_n = np.minimum( (jj+1)*b, n )

            #  the last slice dimensions can be less than a x b
            diff_m = end_m - start_m
            diff_n = end_n - start_n

            im_out[ start_m : end_m, start_n : end_n ] = (
                 slices[ ii, jj, 0:diff_m, 0:diff_n ]
                )

    return im_out