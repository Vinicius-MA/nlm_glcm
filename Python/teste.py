import time
from enum import Enum
from os.path import exists

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.feature import greycomatrix, greycoprops
from skimage.restoration import denoise_nl_means
from skimage.transform import rescale, resize

import nlm_glcm
import utils as ut
from glcm_properties import Props
from nlm_glcm import nlm_glcm_filter
from nlm_lbp import nonlocal_means_lbp_original as nlm_lbp
from noise_sampling import BaseImage
from nonlocal_means import nonlocal_means_original


def synthetic_image( img_path = 'image.jpg'):

    img = np.empty((100,100), dtype=np.uint8)

    for i in range(0,10):
        img[:,10*i:10*(i+1) ] = 255*0.1*i

    io.imsave('Python/testes/imagem_teste.jpg', img, quality=100)

def list2str( list_in ):
    
    list_str = '['
    
    for l in list_in:
        list_str += f'{l:#.02f}'

        if l < len(list_in) - 1:
            list_str += f', '
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

def teste2(img_in_path, test_category, test_number, sigma, props, distances, angles,
    window_radius, patch_radius, symmetric, levels=256, rescale_factor=1.0,
    plot=False, save=True , exec_nlmlbp = False
 ):    

    # folder and output name
    folder = 'Python/testes/'
    im_name = f"teste_{test_category:02d}_{test_number:03d}.jpg"

    print()
    print( 70 * '*' )
    print(
        f"""Test {test_category}/{test_number}:
            \r\tsigma={sigma}
            \r\tprops={props}
            \r\tdistances={distances}
            \r\tangles={list2str(angles)}
            \r\twindow_radius={window_radius}
            \r\tpatch_radius={patch_radius}
            \r\tlevels={levels}
            \r\tsymmetric={symmetric}
        """
    )

    # read image and generate noisy version
    image = io.imread( img_in_path, as_gray=True)
    image = rescale(image, rescale_factor )

    if(image.dtype != np.uint8):
        image = ( (levels-1) * image).astype(np.uint8)    
    
    image_n = ut.add_gaussian_noise( image, sigma=sigma, max_gray=levels-1)
        
    # Non Local Means Algorithm
    t0 = time.time()
    im_nlm = denoise_nl_means(image_n, patch_radius, window_radius, sigma, preserve_range=True)
    dif_nlm = time.time() - t0

    t0 = time.time()
    im_nlmlbp = ( nlm_lbp(image_n, window_radius, patch_radius, sigma, 'default', 8, 1) 
        if exec_nlmlbp else np.zeros_like(image_n))
    dif_lbp = time.time() - t0
    
    # NLM + GLCM proposed algorithm
    if not( exists( folder+im_name ) ):
        
        t0 = time.time()    
        image_out = nlm_glcm_filter(image_n, window_radius, patch_radius, sigma,
            np.array(distances,np.uint8), np.array(angles, np.float64),
            levels, props, symmetric
        )
        dif_glcm = time.time() - t0

        if save: io.imsave(folder+im_name, image_out)

    else:
        dif_glcm = 0
        image_out = io.imread( folder+im_name, as_gray=True )

    # Printing PSNRs and timings
    print('\tPSNR:')
    print( f'\t noisy: { ut.calculate_psnr(image, image_n) }' )
    print( f'\t NLM: {ut.calculate_psnr(image, im_nlm)} in {int(dif_nlm//60)}:{dif_nlm%60:#.02f}')
    if exec_nlmlbp: print( f'\t NLM_LBP: { ut.calculate_psnr(image, im_nlmlbp)} in {int(dif_lbp//60)}:{int(dif_lbp%60):#02d}')
    print( f'\t NLM_GLCM: { ut.calculate_psnr(image, image_out)} in {int(dif_glcm//60)}:{int(dif_glcm%60):#02d}')
    print( 70 * '*' )

    # Plotting
    if plot:
        fig, axes = plt.subplots(2,3)
        fig.suptitle('Teste')
        ax = axes.ravel()
        ax[0].imshow(image[0:100, 0:100], cmap='gray')
        ax[0].set_title('Original')
        ax[1].imshow(image_n[0:100, 0:100], cmap='gray')
        ax[1].set_title('Noisy')
        ax[2].imshow(im_nlm[0:100, 0:100], cmap='gray')
        ax[2].set_title('NLM')
        ax[3].imshow(im_nlmlbp[0:100, 0:100], cmap='gray')
        ax[3].set_title('NLM-LBP') if exec_nlmlbp else ax[3].set_title('Not executed')
        ax[4].imshow(image_out[0:100, 0:100], cmap='gray')
        ax[4].set_title('NLM-GLCM')
        plt.tight_layout()
        plt.show()

def combine_props_2by2():

    img_in_path = 'Python/testes/original.jpg'
    test_category = 1
    test_number = 5
    
    plot = True
    save = True
    exec_nlmlbp = True
    rescale_factor = 1.
    
    sigma = 25
    props = Props.all()
    distances = [ 10 ]
    angles = [ 3*np.pi/4 ]
    window_radius = 10
    patch_radius = 6
    symmetric = True
    levels = 256

    """
    for prop1 in Props:
        for prop2 in Props:
            
            if prop2.value['order'] <= prop1.value['order']:
                continue

            props = [ prop1.value['name'], prop2.value['name'] ]

            teste2( img_in_path, test_category, test_number, sigma, props,
                distances, angles, window_radius, patch_radius, 
                symmetric, levels, False, max_ram_gb
            )

            test_number += 1\
    """

    teste2( img_in_path, test_category, test_number, sigma, props,
        distances, angles, window_radius, patch_radius, 
        symmetric, levels, rescale_factor, plot, save, exec_nlmlbp
    )

def single_test():
    img_in_path = 'image-database/HW_C001_120.jpg'
    test_category = 1000
    test_number = 5000
    
    plot = True
    save = False
    exec_nlmlbp = False
    rescale_factor = 1.
    
    sigma = 25
    props = Props.all()
    distances = [ 10 ]
    angles = [ 0 ]
    window_radius = 10
    patch_radius = 6
    symmetric = False
    levels = 256

    teste2(img_in_path, test_category, test_number, sigma,
        props, distances, angles, window_radius, patch_radius,
        symmetric, levels, rescale_factor, plot, save, exec_nlmlbp
    )

#combine_props_2by2()
#synthetic_image()
single_test()
