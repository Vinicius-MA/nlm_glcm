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

