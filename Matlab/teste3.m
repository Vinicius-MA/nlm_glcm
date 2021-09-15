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

% 13/09 - RODAR ASSIM PARA VER O RESULTADO
NLM_ANDRE = nlm_lbp_andre( In, d, s, h);

k = 1;

Y(k,:,:) = I; tit{k} = 'original'; k = k+1;
Y(k,:,:) = In; tit{k} = 'noisy'; k = k+1;
%Y(k,:,:) = nlm_lbp(I, In, d, s, h); tit{k} = 'nlm_lbp' k = k + 1;
%Y(k,:,:) = nlm(I, In, d, s, h); tit{k} = 'nlm'; k = k+1;
%Y(k,:,:) = nlmfilter(In, d, s, f, h); tit{k} = 'nlmfilter'; k = k+1;
Y(k,:,:) = NLM_ANDRE; tit{k} = 'nlm-lbp-andre'; k = k+1;
%[~, Y(k,:,:)] = nlm_lbp(I, In, d, s, h); tit{k} = 'nlm_lbp'; k = k+1;

K = uint8( size(Y, 1) );

L1 = double( idivide( K, 2) ) ;
L2 = double( idivide( K, 2) + rem(K, 2) );

if L1 == 1
    L1 = L1 + 1;
else
    if L2 == 1
        L2 = L2 + 1;
    end
end

PSNR = zeros( K );

figure,
for n = 1: (k-1)
    
    im(:,:) = Y(n,:,:);
    
    subplot(L1, L2, n), imshow(im), title( tit{n} );
    
    if im == I
        PSNR(n) = NaN ;
        continue;
    end
    
    PSNR(n) = psnr( im, I);
    
    fprintf('psnr(%.0f): %.04f\r\n', n, PSNR(n) );
    
end