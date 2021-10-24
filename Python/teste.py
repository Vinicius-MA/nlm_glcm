import time

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.feature import greycomatrix, greycoprops
from skimage.restoration import denoise_nl_means
from skimage.transform import rescale

from nlm_glcm import nlm_glcm_filter
import nlm_glcm
from noise_sampling import BaseImage
import utils as ut
import chime

def list2str( list_in ):
    
    list_str = '['
    
    for l in list_in:
        list_str += f'{l:#.02f}'

        if l < len(list_in)-2:
            list_str += f','
    list_str += ']'

    return list_str


def teste_noise_sampling():

    sigma_list = [10, 25, 50]

    image_in_folder = 'Python/testes/'
    image_out_folder = 'Python/testes/'
    spreadsheet_folder = 'Python/testes/'

    filenames = ['original.jpg']

    for fname in filenames:

        base_image = BaseImage( f'{fname}', sigma_list, folder=image_in_folder)

        start = time.time()
        base_image.generate_noisy_samples(folder = image_out_folder)
        diff = time.time() - start

        base_image.generate_nlmLbp_samples(folder = image_out_folder)
        base_image.generate_spreadsheet( folder = spreadsheet_folder)

        print( f'>>>> total {fname} time: {diff:#.01f} s ({diff/60:#.01f} min)')

def teste1():

    dists = [ 1, 3, 9, 15 ]
    #dists = [ 5 ]

    #angles = [0]
    angles = [0, np.pi/4, np.pi/2, np.pi]

    plot_shape = [ len(dists), len(angles) ]

    img = ( 255 * io.imread('original.jpg', as_gray=True) ).astype( np.uint8 )

    g = greycomatrix(img, dists, angles, 256)

    features = greycoprops(g)
    print( features.shape )
    print( features )

    fig, ax = plt.subplots( plot_shape[0], plot_shape[1] )

    k=0
    for i in range( plot_shape[0] ):
        for j in range( plot_shape[1] ):

            ax[i,j].imshow( g[ :, :, i, j ] )
            ax[i, j].set_title( f'angle={angles[i]:#.01f},dist={dists[j]*180:#.01f}' )
            k += 1

    plt.show()

def teste2():

    im_name = 'teste078.jpg'

    prop = 'ASM'
    
    distances = [ 10 ]
    angles = [ 0 ]

    window_radius = 10
    patch_radius = 6

    levels = 256
    symmetric = True

    h = 25

    image = io.imread( 'Python/testes/original.jpg', as_gray=True)
    #image = rescale( image, 0.25, anti_aliasing=True)
    image = ( (levels-1) * image).astype(np.uint8)
    image_n = ut.add_gaussian_noise( image, sigma=h, max_gray=levels-1)

    print(f'{im_name} -->\t{h}\t{prop}\t{list2str(angles)}\t{distances}\t{patch_radius}\t{window_radius}\t{symmetric}')

    t0 = time.time()

    image_out = nlm_glcm_filter(image_n, window_radius, patch_radius, h,
        np.array(distances,np.uint8), np.array(angles, np.float64),
        levels, symmetric, prop, 50
    )

    dif = time.time() - t0

    print('PSNR:')
    print( f'\tnoisy: { ut.calculate_psnr(image, image_n) }' )
    print( f'\tfiltered: { ut.calculate_psnr(image, image_out)}')

    print( f'nlm_glcm_filter total elapsed time: {int(dif//60)} min {dif%60:#.02f} s')

    io.imsave(f'Python/testes/{im_name}', image_out)

    for i in range(10):
        chime.error()
        time.sleep(0.5)

    fig, axes = plt.subplots(1,3)
    ax = axes.ravel()

    ax[0].imshow(image[0:100, 0:100], cmap='gray')
    ax[0].set_title('Original')

    ax[1].imshow(image_n[0:100, 0:100], cmap='gray')
    ax[1].set_title('Noisy')

    ax[2].imshow(image_out[0:100, 0:100], cmap='gray')
    ax[2].set_title('Output')

    plt.tight_layout()
    plt.show()

def teste_dev():

    distances = np.array( [5] )
    angles = np.array( [0., np.pi/2], dtype=np.float64 )

    window_radius = 10
    patch_radius = 6

    h = 25

    hw = 6
    w = 2*hw + 1

    levels = 64

    image = io.imread( 'Python/testes/original.jpg', as_gray=True)
    #image = rescale( image, 0.25, anti_aliasing=True)
    image = ( (levels-1) * image).astype(np.uint8)
    slices = ut.image2slices(image, 8, 4)
    im_out = ut.slices2image(image, slices)

    print( False not in ( im_out == image ) )

    fig, axes = plt.subplots(1,2)
    ax = axes.ravel()

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original')

    ax[1].imshow(im_out, cmap='gray')
    ax[1].set_title('Out')

    plt.tight_layout()
    plt.show()

teste2()
#teste_dev()