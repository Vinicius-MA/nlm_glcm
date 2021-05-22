%%
% path to the mex files
addpath('mex');
% if OpenMP is not installed 
%addpath('mex_no_OMP');

% Read image and add noise
img = imread('test.png');
stddev = 30.;

noise = stddev * randn(size(img));
noisy = double(img) + noise;
figure(1);
imshow(uint8(noisy));

%% 
% Use NLDD using Matlab data types
% This procedure receives a noisy image and denoises it using NLDD
resNLDD = NLDD(noisy, stddev, 1);
rmseNLDD = sqrt(mean((resNLDD(:) - double(img(:))).^2));
psnrNLDD = 20 * log10((255)/ rmseNLDD);
figure(2);
imshow(uint8(resNLDD));
%%
% NlBayes Denoiser
resNLBayes = NlBayesDenoiser(noisy, stddev, 1);
rmseNlBayes = sqrt(mean((resNLBayes(:) - double(img(:))).^2));
psnrNlBayes = 20 * log10((255)/ rmseNlBayes);
figure(3);
imshow(uint8(resNLBayes));
%%
% Use DDID step with NLBayes result
resMatlabNLDD = DDIDstep(resNLBayes,noisy,stddev^2, 15, 7, 0.7, 0.8);
figure(4);
imshow(uint8(resMatlabNLDD));
rmseMatlabNLDD = sqrt(mean((resMatlabNLDD(:) - double(img(:))).^2));
psnrMatlabNLDD = 20 * log10((255)/ rmseMatlabNLDD);
%% 
% DDID step denoiser
y = noisy;
x1 = DDIDstep(y,y,stddev^2, 15, 7, 100, 4.0);
figure(5);
imshow(uint8(x1));
x2 = DDIDstep(x1,y,stddev^2, 15, 7, 8.7, 0.4);
figure(6);
imshow(uint8(x2));
x3 = DDIDstep(x2,y,stddev^2, 15, 7, 0.7, 0.8);
figure(7);
imshow(uint8(x3));
rmseDDID = sqrt(mean((x3(:) - double(img(:))).^2));
psnrDDID = 20 * log10((255)/ rmseDDID);
