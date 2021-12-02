import time

import numpy as np
import utils as ut
import skimage.io as io
from noise_sampling import _imread
from skimage.metrics import structural_similarity as ssim
from nlm_glcm import nlm_glcm_filter
from os.path import exists
from glcm_properties import Props

def test_exec_nlmglcm_with_param(teste_num, fname="HW_C009_120.jpg",
     sigma=25, window_radius=10, patch_radius=6, distances=[10],
     angles=[0.], symmetric=True, props=Props.all()
    ):

    # static
    
    sample=1
    # folders
    folder_orig = "../image-database/"
    folder_noisy = "../images-noisy/"
    folder_save = "testes-parameters/"
    # file names
    fname_orig = folder_orig + fname
    fname_noisy = folder_noisy + fname.replace(".jpg", f"_sigma{sigma:03d}_{sample:02d}_noisy.jpg" )
    fname_save = folder_save + fname.replace(
        ".jpg", f"_sigma{sigma:03d}_{sample:02d}_teste{teste_num:03d}.tif"
    )
    # print start
    print(">>>> teste_exec_nlmglcm_with_param:")
    print( f"TESTE_NUM={teste_num}")
    print( f"\tinput file name: {fname}")
    print( f"\tsigma: {sigma}" )
    print( f"\twindow_radius={window_radius}")
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
        print( f"\t<-- file read: {fname_save}")
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
    distances = [2, 3, 5, 7, 10]
    symmetric = [True, False]
    teste_num = 1

    # T[1~8] variando ângulo e simetria da GLCM
    for theta in angles:
        for symm in symmetric:
            test_exec_nlmglcm_with_param(teste_num, angles=[theta], symmetric=symm)
            teste_num += 1
    # T[9~14] variando propriedades utilizadas (append)
    props = []
    for pp in Props.all():
        props.append(pp)
        test_exec_nlmglcm_with_param(teste_num, props=props)
        teste_num+=1
    # T[15~20] variando propriedades utilizadas (um a uma)
    for pp in Props.all():
        test_exec_nlmglcm_with_param(teste_num, props=[pp])
        teste_num+=1
    # T[21] removendo propriedades com pior desempenho (1,4)
    props = []
    for pp in Props.all():
        if pp not in Props.get_list([Props.DISSIMILARITY, Props.CORRELATION]):
            props.append(pp)
    test_exec_nlmglcm_with_param(teste_num, props=props)
    teste_num += 1
    # T[22] removendo propriedades com pior desempenho (0,1,4)
    props = []
    for pp in Props.all():
        if pp not in Props.get_list([Props.CONTRAST,Props.DISSIMILARITY, Props.CORRELATION]):
            props.append(pp)
    test_exec_nlmglcm_with_param(teste_num, props=props)
    teste_num += 1
    # T[23~32] variando distância da GLCM e simetria (theta=3pi/4, symm=false, props=2)
    theta = angles[3]
    props = Props.get_list([Props.HOMOGENEITY])
    for dd in distances:
        for sym in [True, False]:
            test_exec_nlmglcm_with_param( teste_num,
                distances=[dd], angles=[theta], symmetric=sym,props=props )
            teste_num += 1    
    # T[33~44] variando imagem filtrada (sigma=25)
    image_indexes = [1,2,4,9,11,12,13,15,16,17,19,24]
    sigma = 25
    dist = distances[4]
    theta = angles[3]
    sym = True
    props = Props.get_list([Props.HOMOGENEITY])
    for index in image_indexes:
        fname= f"HW_C{index:03d}_120.jpg"
        test_exec_nlmglcm_with_param(teste_num, fname=fname, sigma=sigma,
            distances=[dist], angles=[theta], symmetric=sym, props=props)
        teste_num += 1
    # T[45~56] variando imagem filtrada (sigma=50)
    image_indexes = [1,2,4,9,11,12,13,15,16,17,19,24]
    sigma = 50
    dist = distances[4]
    theta = angles[3]
    sym = True
    props = Props.get_list([Props.HOMOGENEITY])
    for index in image_indexes:
        fname= f"HW_C{index:03d}_120.jpg"
        test_exec_nlmglcm_with_param(teste_num, fname=fname, sigma=sigma,
            distances=[dist], angles=[theta], symmetric=sym, props=props)
        teste_num += 1

main()
