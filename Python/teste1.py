import time

import cv2 as cv
import skimage
import skimage.restoration
from matplotlib import pyplot as plt

import myIPL as ipl

img_path = 'HW_C001_000.jpg'

std = [10, 25, 50]
var = (std[1] / (2**8-1)) ** 2
h = std[1]

I = cv.imread(img_path)
Im = cv.cvtColor(I, cv.COLOR_BGR2GRAY)
Im = ipl.imresize(Im, 0.7)

In = skimage.img_as_ubyte(skimage.util.random_noise(Im, mode='gaussian', var=var))
Y1 = skimage.restoration.denoise_nl_means(In, h = std[1], fast_mode=False, sigma=h)#, preserve_range=True)
Y2 = ipl.nlm_lbp(In, 10, 6, h, P=8, R=1)
#Y3 = ipl.nlm(In, 10, 6, h)
#Y4 = ipl.nlm_andre(In, 10, 6, 10*h)
t0 = time.time()
Y5 = ipl.denoise_nl_means(I, h=std[1], sigma=h)
print(f'time: {time.time() - t0}')

__ = plt.subplot(2,2,1), plt.imshow(Im, 'gray'), plt.title('Original')
__ = plt.subplot(2,2,2), plt.imshow(In, 'gray'), plt.title('Noisy')
__ = plt.subplot(2,2,3), plt.imshow(Y1, 'gray'), plt.title('NLM')
#plt.subplot(2,2,4), plt.imshow(Y2, 'gray'), plt.title('NLM-LBP')
#plt.subplot(2,2,4), plt.imshow(Y3, 'gray'), plt.title('NLM')
__ = plt.subplot(2,2,4), plt.imshow(Y5, 'gray'), plt.title('NLM-André')
plt.show()
