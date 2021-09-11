function [ u1, u2 ] = nlm_lbp(u, v, srch_r, sim_r, h, g, fdist, fweight)
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

glbp = ones(2*sim_r+1, 2*sim_r+1);
 
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

% LBP WEIGHT

u_ = padarray(u, [1 1], 'symmetric');
%v_ = padarray(v, [1 1], 'symmetric');

SP=[-1 -1; -1 0; -1 1; 0 -1; -0 1; 1 -1; 1 0; 1 1];
u_lbp = double ( lbp( u_, SP, 0, 'i' ) ) / 256;
%v_lbp = double ( lbp( v_, SP, 0, 'i' ) ) / 256;

u_lbp_pad = padarray(u_lbp, [srch_r, srch_r], 'symmetric');

w_lbp_max = zeros(m,n);
w_lbp_sum = zeros(m,n);
%w_m = zeros(m,n);

% search window shifts
[X, Y] = meshgrid(-srch_r:srch_r, -srch_r:srch_r);
% search window size
N = (2*srch_r + 1)^2;

for i = 1:N
    
    tic
    
    if X(i) == 0 && Y(i) == 0
        continue; 
    end
    
    % LBP WEIGHT
    u_lbp_i = u_lbp_pad(srch_r + (1:m) + Y(i), srch_r + (1:n) + X(i));
    dlbp = imfilter( chi_sqr_diss( u_lbp, u_lbp_i, 6 ), glbp, 'symmetric' );
    hSi = std( std( dlbp) );
    w_lbp = fweight(dlbp, hSi);
    w_lbp_max = max(w_lbp_max, w_lbp);
    w_lbp_sum = w_lbp_sum + w_lbp;
    
    % shifted noise free image
    ui = u_pad(srch_r + (1:m) + Y(i), srch_r + (1:n) + X(i));
    % shifted noisy image
    vi = v_pad(srch_r + (1:m) + Y(i), srch_r + (1:n) + X(i));
    % compute weighted euclidean distance
    d = imfilter(fdist(u, ui), g, 'symmetric');
    % compute weights 
    w = fweight(d,h);
    % update max weight
    w_max = max(w_max, w);
    % update wieght sum
    w_sum = w_sum + w;    
    % update weighted average
    u1 = u1 + w.*vi;
    
    u2 = u2 + (w .* w_lbp) .* vi;
    
    %w_m = w .* w_lbp;
    
    t = toc;
    
    fprintf('pixel %d  processed in %.03f\r\n', i, t);
    
end

%LBP WEIGHT
w_lbp_max(w_lbp_max == 0) = 1;
w_lbp_sum = w_lbp_sum + w_lbp_max;

% avoid division by zero
w_max(w_max == 0) = 1;
% weight fix: max weight is assigned to the center samples
u1 = u1 + w_max.*v;
% add the weight contribution of the center pixels
w_sum = w_sum + w_max;
% normalize results
u1 = u1./w_sum;

%w_m = w_m ./ (w_sum .* w_lbp_sum);

u2 = u2./(w_sum .* w_lbp_sum);
%u2 = u1 + w_m .* v;

end

