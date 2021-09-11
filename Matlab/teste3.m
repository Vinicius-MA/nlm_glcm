clear all
close all
clc

sigma = 25;
v = (sigma/(2.^8-1)).^2;

I = rgb2gray(imread('x.jpg'));
In = imnoise(I, 'gaussian', 0, v);

I = double(I)/max(double(I(:)));
In = double(In)/max(double(In(:)));

SP=[-1 -1; -1 0; -1 1; 0 -1; -0 1; 1 -1; 1 0; 1 1];
I_lbp = lbp(In,SP,0,'i');
I_lbp = double(I_lbp)/256;

d = 10;
s = 6.0;
h = v^0.5;
f = @(x1, x2) (x1 - x2).^2;

k = 1;

Y(k,:,:) = I; tit(k, :) = 'noisy'; k = k+1;
%Y(k,:,:) = nlm_lbp(I, In, d, s, h); tit(k, :) = 'nlm_lbp' k = k + 1;
%Y(k,:,:) = nlm(I, In, d, s, h); tit(k, :) = 'nlm'; k = k+1;
%Y(k,:,:) = nlmfilter(In, d, s, f, h); tit(k, :) = 'nlmfilter'; k = k+1;
Y(k,:,:) = nlm_lbp_andre( In, d, s, h); tit(k, :) = 'lnm_lbp_andre'; k = k+1;
%[~, Y(k,:,:)] = nlm_lbp(I, In, d, s, h); tit(k, :) = 'nlm_lbp'; k = k+1;

K = uint8( size(Y, 1) );

L1 = idivide( K, 2) ;
L2 = idivide( K, 2) + rem(K, 2);

figure,
for n = 1: k-1
    subplot(L1, L2, n), imshow( Y(k,:,:) ), title( tit(k,:) );
    psnr(n) = psnr( Y(k,:,:), I);
end
