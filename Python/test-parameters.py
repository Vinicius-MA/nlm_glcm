import time

import numpy as np
import utils as ut
import skimage.io as io
from noise_sampling import _imread
from skimage.metrics import structural_similarity as ssim
from nlm_glcm import nlm_glcm_filter
from os.path import exists
from glcm_properties import Props

def test_exec_nlmglcm_with_param(teste_num, window_radius=10, patch_radius=6,
     distances=[10], angles=[0.], symmetric=True, props=Props.all()
    ):

    # static
    sigma=25
    sample=1
    # folders
    folder_orig = "../image-database/"
    folder_noisy = "../images-noisy/"
    folder_save = "testes-parameters/"
    # file names
    fname="HW_C009_120.jpg"
    fname_orig = folder_orig + fname
    fname_noisy = folder_noisy + fname.replace(".jpg", f"_sigma{sigma:03d}_{sample:02d}_noisy.jpg" )
    fname_save = folder_save + fname.replace(
        ".jpg", f"_sigma{sigma:03d}_{sample:02d}_teste{teste_num:03d}.jpg"
    )
    # print start
    print(">>>> teste_exec_nlmglcm_with_param:")
    print( f"TESTE_NUM={teste_num}")
    print( f"\t window_radius={window_radius}")
    print( f"\tpatch_radius={patch_radius}")
    print( f"\tdistances = {distances}")
    print( f"\tangles = {ut.list2str(angles)}")
    print( f"\tsymmetric={symmetric}")
    print( f"\tprops={props}")
    # open original file
    im_orig = _imread( fname_orig )
    print( f"\t<-- im_orig read: {fname_orig}")
    # open noisy file    
    im_noisy = _imread( fname_noisy)
    print( f"\t<-- im_noisy read: {fname_noisy}")
    # execute NLM-GLCM
    str_time = ""
    if exists(fname_save):
        im_out = _imread(fname_save)
    else:
        t0 = time.time()
        im_out = nlm_glcm_filter( im_noisy, window_radius, patch_radius,
            sigma, distances, angles, symmetric=symmetric, props=props
        )
        dif = time.time() - t0
        str_time = f"\t--> time: {int(dif//60):02d}:{int(dif%60):02d}"
    # save output file
    io.imsave(fname_save, im_out)
    print( f"\t--> file saved: {fname_save}")
    # calculates and print SSIM, PSNR
    values_noisy = [ut.calculate_psnr(im_orig, im_noisy), ssim(im_orig, im_noisy)]
    values_out = [ ut.calculate_psnr(im_orig, im_out), ssim(im_orig, im_out)]
    print( f"\t*noisy: {fname_noisy}\n\t\tpsnr: {values_noisy[0]:f}\t\tssim: {values_noisy[1]:f}")
    print( f"\t*out: {fname_save}\n\t\tpsnr: {values_out[0]:f}\t\tssim: {values_out[1]:f}")
    print( str_time )

def main():

    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    symmetric = [True, False]
    teste_num = 1

    # variando ângulo e simetria da GLCM
    for theta in angles:
        for symm in symmetric:
            test_exec_nlmglcm_with_param(teste_num, angles=[theta], symmetric=symm)
            teste_num += 1
    # variando propriedades utilizadas (append)
    props = []
    for p in Props.all():
        props.append(p)
        test_exec_nlmglcm_with_param(teste_num, props=props)
        teste_num+=1
    # variando propriedades utilizadas (um a uma)
    for p in Props.all():
        test_exec_nlmglcm_with_param(teste_num, props=[p])
        teste_num+=1

main()
