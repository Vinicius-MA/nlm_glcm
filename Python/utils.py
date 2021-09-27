import numpy as np
import numba
from numba import jit,njit, float64
import math

@njit()
def normalize_data(data,eps):
    return (data - np.min(data)) / (np.max(data) - np.min(data) + eps)

@njit()
def stein(s,sigma):
    return np.exp(-s/sigma)

@njit()
def chisquare_distance(A, B, eps): 
    return np.sum( (A-B)**2 / (A+B+eps) )


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
