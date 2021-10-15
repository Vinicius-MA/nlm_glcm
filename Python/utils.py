"""
    Credits: Gregory Petterson Zanelato <gregory.zanelato@usp.br>

    References:
        https://scikit-image.org/
"""

import math

import numba
import numpy as np
from numba import float64, jit, njit

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

def add_gaussian_noise(image, mean=0, sigma=20):
    """Add Gaussian noise to an image of type np.uint8."""
    gaussian_noise = np.random.normal(mean, sigma, image.shape)
    gaussian_noise = gaussian_noise.reshape(image.shape)
    noisy_image = image + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255)
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

@njit()
def graycomatrix(image, distances, angles, levels, symmetric, normed):

    G = np.zeros( (np.uint16(levels), np.uint16(levels), np.uint16(len(distances)),np.uint16(len(angles)) ), dtype=np.uint32 )

    # count co-occurrences
    glcm_loop(image, distances, angles, levels, G)

    # make G symmetric
    if symmetric:
        Gt = np.transpose( G, (1, 0, 2, 3) )
        G += Gt
    
    # normalize G
    if normed:
        Gn = np.zeros( G.shape, dtype=np.float64 )
        Gn = G.astype( np.float64 )
        glcm_sums = np.sum( Gn, axis=(0,1) )#, keepdims=True)
        for elem in np.nditer(glcm_sums):
            if elem == 0:
                elem = 1
        Gn /= glcm_sums

    return G

@njit()
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
            for r in range( start_row, end_row ):
                for c in range( start_col, end_col ):
                    i = np.uint16( image[r, c] )
                    # computes pixel position
                    j_row = r + offset_row
                    j_col = c + offset_col
                    # comparing pixel
                    j = np.uint16( image[ j_row, j_col ] )
                    if ( i >= 0 and i< levels) and ( j >= 0 and j < levels ):
                        G[i, j, d_idx, a_idx] += 1

@njit()
def graycoprops(G, prop='contrast'):
    
    (num_level, num_level2, num_dist, num_angle) = G.shape

    # normalize GLCM
    G = G.astype( np.float64 )
    glcm_sums = np.sum( G, axis=(0, 1), keepdims=True )
    for elem in np.nditer(glcm_sums):
        if elem == 0:
            elem = 1
    G /= glcm_sums

    # create weights for specified properties
    #I, J = np.ogrid[0:num_level, 0:num_level]
    lSpace = np.linspace(0 ,num_level-1, num=num_level)
    J, I = np.meshgrid( lSpace, lSpace, sparse=True )
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
        #raise ValueError('invalid property: ', prop)
    
    # compute property for each GLCM
    if prop == 'energy':
        asm = np.sum(G ** 2, axis=(0, 1))
        results = np.sqrt(asm)
    elif prop == 'ASM':
        results = np.sum(G ** 2, axis=(0, 1))
    elif prop == 'correlation':
        results = np.zeros((num_dist, num_angle), dtype=np.float64)
        I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
        J = np.array(range(num_level)).reshape((1, num_level, 1, 1))
        diff_i = I - np.sum(I * G, axis=(0, 1))
        diff_j = J - np.sum(J * G, axis=(0, 1))

        std_i = np.sqrt(np.sum(G * (diff_i) ** 2, axis=(0, 1)))
        std_j = np.sqrt(np.sum(G * (diff_j) ** 2, axis=(0, 1)))
        cov = np.sum(G * (diff_i * diff_j), axis=(0, 1))

        # handle the special case of standard deviations near zero
        mask_0 = std_i < 1e-15
        mask_0[std_j < 1e-15] = True
        results[mask_0] = 1

        # handle the standard case
        mask_1 = ~mask_0
        results[mask_1] = cov[mask_1] / (std_i[mask_1] * std_j[mask_1])
    elif prop in ['contrast', 'dissimilarity', 'homogeneity']:
        weights = weights.reshape((num_level, num_level, 1, 1))
        results = np.sum(G * weights, axis=(0, 1))

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
