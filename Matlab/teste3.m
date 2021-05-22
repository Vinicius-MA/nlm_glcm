clear all
close all
clc

sigma = 25;
v = (sigma/(2.^8-1)).^2;

I = rgb2gray(imread('HW_C001_000.jpg'));
In = imnoise(I, 'gaussian', 0, v);

I = double(I)/max(double(I(:)));
In = double(In)/max(double(In(:)));

d = 10;
s = 6.0;
h = v^0.5;
f = @(x1, x2) (x1 - x2).^2;

Y = nlm_lbp(I, In, d, s, h);
%Y2 = nlmfilter(In, d, s, f, h);

figure,
subplot(2,2,1), imshow(I), title('Original');
subplot(2,2,2), imshow(In), title('Noisy');
subplot(2,2,3), imshow(Y), title('NLM');
%subplot(2,2,4), imshow(Y2), title('NLM-André');

psnr_n = psnr(In, I);
psnr_nlm = psnr(Y, I);