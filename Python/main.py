from nonlocal_means import nonlocal_means_original
from nlm_lbp import nonlocal_means_lbp_original
import utils
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

file_name = '../original.jpg'
std = 25

img =  ( 255 * io.imread(file_name, as_gray=True) ).astype(np.uint8)
noisy = utils.add_gaussian_noise(img, 0, std)
psnr_noisy = psnr(img, noisy)
ssim_noisy = ssim(img, noisy)

window_radius = 10 # 21x21
path_radius = 6    # 13x13
h = std

#NLM ORIGINAL
nlm_original = nonlocal_means_original(input=noisy,
                                       window_radius=window_radius,
                                       patch_radius=path_radius,
                                       h=h)
psnr_nlm_original = psnr(img, nlm_original)
ssim_nlm_original = ssim(img, nlm_original)

# NLM + LBP
lbp_method = 'uniform'
lbp_n_points = 16
lbp_radius = 2 
nlm_lbp = nonlocal_means_lbp_original(input=noisy,
                            window_radius=window_radius,
                            patch_radius=path_radius,
                            h=h,
                            lbp_method=lbp_method,
                            lbp_n_points=lbp_n_points,
                            lbp_radius=lbp_radius)
psnr_nlm_lbp = psnr(img, nlm_lbp)
ssim_nlm_lbp = ssim(img, nlm_lbp)


fig = plt.figure()

fig.add_subplot(2, 2, 1)
plt.title('ORIGINAL', fontsize=8)
plt.imshow(img,cmap="gray") 

fig.add_subplot(2, 2, 2)
plt.title('NOISY - PSNR:{:.2f},SSIM:{:.2f}'.format(psnr_noisy, ssim_noisy), fontsize=8)
plt.imshow(noisy,cmap="gray") 

fig.add_subplot(2, 2, 3)
plt.title('NLM. - PSNR:{:.2f},SSIM:{:.2f}'.format(psnr_nlm_original, ssim_nlm_original), fontsize=8)
plt.imshow(nlm_original,cmap="gray")

fig.add_subplot(2, 2, 4)
plt.title('NLM LBP. - PSNR:{:.2f},SSIM:{:.2f}'.format(psnr_nlm_lbp, ssim_nlm_lbp), fontsize=8)
plt.imshow(nlm_lbp,cmap="gray")

plt.show()