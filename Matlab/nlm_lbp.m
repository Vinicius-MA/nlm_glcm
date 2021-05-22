function [ u1, u2 ] = nlm(u, v, srch_r, sim_r, h, g, fdist, fweight)
%NLM
%   Performs NLM denoising
%
%PARAMETRES
%   u       : noise free image
%   v       : noisy image
%   srch_r  : search range
%   sim_r   : similarity range
%   h       : filtering parameters
%   g       : similarity kernel
%   fdist   : similarity measure 
%   fweight : weight function
%
%RETURNS
%   u1      : weight means
%   u2      : weight variances
%
%REFERENCES
%
%   [1] Antoni Buades, Bartomeu Coll, Jean-Michel Morel. A review of image 
%       denoising algorithms, with a new one. SIAM Journal on Multiscale 
%       Modeling and Simulation: A SIAM Interdisciplinary Journal, 2005, 
%       4 (2), pp.490-530. <hal-00271141>
%
%   [2] L. Condat, “A simple trick to speed up and improve the non-local 
%       means,” research report hal-00512801, Aug. 2010, Caen, France.
%       Unpublished.
%
%   [3] A. A. Bindilatti and N. D. A. Mascarenhas, “A Nonlocal Poisson 
%       Denoising Algorithm Based on Stochastic Distances,” 
%       IEEE Signal Processing Letters, vol. 20, no. 11, pp. 1010–1013, 
%       Nov. 2013.
%

% if fdist is not specified
if ~exist('fdist','var')
    fdist = @(a, b) (a - b).^2; % use euclidean distance
end

% if fweight is not specified
if ~exist('fweight','var')
    fweight = @(d,h) exp(-d./h); % use nlm weight function
end

% if g is not specified
if ~exist('g','var')
    %g = disk_kernel(sim_r);
    g = fspecial('disk', sim_r);
end
 
% image size
[m, n] = size(u);
% output
u1 = zeros(m,n);
u2 = zeros(m,n);
% pads noise free image 
u_pad = padarray(u, [srch_r srch_r], 'symmetric');
% pads noisy image 
v_pad = padarray(v, [srch_r srch_r], 'symmetric');
% max weight
w_max = zeros(m,n);
% weight sum
w_sum = zeros(m,n);
% squared h
h = h.*h;

% search window shifts
[X, Y] = meshgrid(-srch_r:srch_r, -srch_r:srch_r);
% search window size
N = (2*srch_r + 1)^2;

%   LBP (R=1, P=8)
lbp_t = srch_r;
hl = 6;
SP = [-1 -1; -1 0; -1 1; 0 -1; -0 1; 1 -1; 1 0; 1 1];
v_lbp = double(lbp(padarray(u, [1 1],'symmetric'), SP, 0, 'i'));
lbp_patch = im2patch(v_lbp, lbp_t + hl);

w_lbp = zeros( size(lbp_patch) );
for r = 1:m
    for s = 1:n
        
        P = lbp_patch(r,s , :, :);
        w_lbp(r, s, :, :) = lbp_weight(P, hl);
        
    end
end

[X_lbp, Y_lbp] = meshgrid(-(lbp_t):(lbp_t), -(lbp_t):(lbp_t));

for i = 1:N
    
    if X(i) == 0 && Y(i) == 0
        continue; 
    end
    
    % shifted noise free image
    ui = u_pad(srch_r + (1:m) + Y(i), srch_r + (1:n) + X(i));
    % shifted noisy image
    vi = v_pad(srch_r + (1:m) + Y(i), srch_r + (1:n) + X(i));
    % compute weighted euclidean distance
    d = imfilter(fdist(u, ui), g, 'symmetric');
    % compute weights 
    w = fweight(d,h);
    % update weighted average
    u1 = u1 + w.*vi;
    % update weighted variance
    u2 = u2 + w.*(vi.^2);
    % update max weight
    w_max = max(w_max, w);
    % update wieght sum
    w_sum = w_sum + w;
    
    Si = lbp_pad(lbp_t + (1:(2*lbp_t+1)) + Y_lbp(i), lbp_t + (1: (2*lbp_t+1)) + X_lbp(i) );
    % compute weighted Chi-square dissimilarity
    Si_weighted = Si .* lbp_weight(lbp_pad, Si, lbp_t, hl);
    
    lbp_result(i) = sum(sum(Si_weighted));
    
end

% avoid division by zero
w_max(w_max == 0) = 1;

% weight fix: max weight is assigned to the center samples
u1 = u1 + w_max.*v;
u2 = u2 + w_max.*(v.^2);

% add the weight contribution of the center pixels
w_sum = w_sum + w_max;

% normalize results
u1 = u1./w_sum;
u2 = u2./w_sum;

% nonlocal variance
u2 = u2 - u1.^2;

end

function d_lbp = lbp_distance(Li, Lj)% Chi-square dissimilarity metric

Hi = imhist( uint8(Li) );
Hj = imhist( uint8(Lj) );

d_lbp = sum( (Hi - Hj).^2 ./ (1e-9 + Hi + Hj) );


end


function w_lbp = lbp_weight(patch, hl)

[M, N] = size(patch);

w_lbp = zeros(M,N);

xi = floor(N/2) + 1;
yi = floor(M/2) + 1;

K = M*N;

Li = patch(xi + (-hl:+hl), yi + (-hl:+hl) );
Hi = imhist( uint8(Li) );

[X, Y] = meshgrid(-N:N, -M:M);

for j = 1:K
    
    if X(j) == 0 && Y(j) == 0
        continue; 
    end
    
    xj = xi + X(j);
    yj = yi + Y(j);
    
    Lj = patch(xj + (-hl:+hl), yj + (-hl:+hl));
    Hj = imhist(Lj);
    
    % stores dissimilarity into w array
    w_lbp(k) = (Hi - Hj).^2 / (Hi + Hj + 1e-9);
    
end

hSi = std( std(w_lbp) );

w_lbp = exp(-w_lbp/hSi);

Zi = sum( sum(w_lbp) );

w_lbp = w_lbp / Zi;

end

% N_lbp = (2*(lbp_t)+1)^2;
% 
% d_lbp = zeros(M,N);
% 
% for j=1:N_lbp
%     
%     if X_lbp(j) == 0 && Y_lbp(j) == 0
%         continue; 
%     end
%     
%     Lj = lbp_pad( floor(M/2) + X_lbp(j) + (1:(2*lbp_l_half+1)), floor(N/2) + Y_lbp(j) + (1:(2*lbp_l_half+1) ) );
%     Hj = imhist( uint8(Lj) );
%     
%     d_lbp(j) = sum( (Hi - Hj).^2 ./ (1e-9 + Hi + Hj) );
% end
% 
% % region smoothness estimation
% hSi = std(std(d_lbp));
% 
% % normalizing dissimilarities
% d_lbp = d_lbp./hSi;
% 
% w = exp(-d_lbp);
% 
% w_sum = sum(sum(w));
% 
% w = w / w_sum;