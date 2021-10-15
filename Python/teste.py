import time

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.feature import greycomatrix, greycoprops

from nlm_glcm import nlm_glcm_filter
from noise_sampling import BaseImage
from utils import *


def teste_noise_sampling():

    sigmaList = [10, 25, 50]
    
    imageInFolder = 'Python/testes/'
    imageOutFolder = 'Python/testes/'
    spreadsheetFolder = 'Python/testes/'

    filenames = [f'original.jpg']

    for fname in filenames:
    
            baseImage = BaseImage( f'{fname}', sigmaList, folder=imageInFolder)

            start = time.time()
            baseImage.generate_noisy_samples(folder = imageOutFolder)
            diff = time.time() - start
            
            baseImage.generate_nlmLbp_samples(folder = imageOutFolder)
            baseImage.generate_spreadsheet( folder = spreadsheetFolder)

            print( f'>>>> total {fname} time: {diff:#.01f} s ({diff/60:#.01f} min)')

def teste1():

    dists = [ 1, 3, 9, 15 ]
    #dists = [ 5 ]

    #angles = [0]
    angles = [0, np.pi/4, np.pi/2, np.pi]

    plotShape = [ len(dists), len(angles) ]

    img = ( 255 * io.imread('original.jpg', as_gray=True) ).astype( np.uint8 )

    g = greycomatrix(img, dists, angles, 256)

    features = greycoprops(g)
    print( features.shape )
    print( features )

    fig, ax = plt.subplots( plotShape[0], plotShape[1] )

    k=0
    for i in range( plotShape[0] ):
        for j in range( plotShape[1] ):
            
            ax[i,j].imshow( g[ :, :, i, j ] )
            ax[i, j].set_title( f'angle={angles[i]:#.01f},dist={dists[j]*180:#.01f}' )
            k += 1

    plt.show()

def teste2():

    dists = [5]
    angles = [0]

    window_radius = 10
    patch_radius = 6

    h = 25

    levels = 256

    image = ( 255 * io.imread( 'Python/original.jpg', as_gray=True ) ).astype( np.uint8 )
    image_n = add_gaussian_noise( image, sigma=h )

    image_out = nlm_glcm_filter( image_n, window_radius, patch_radius, h, dists, angles, levels, False, True )

    print( f'PSNR:\n\tnoisy: {calculate_psnr(image, image_n)}\n\tfiltered: {calculate_psnr(image, image_out)}')

    plt.imshow(image_out)
    plt.show()

teste2()
