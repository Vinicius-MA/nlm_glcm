function [psnr]=psnr(A, B, MAX)

 if A == B
    error('Imagens são identicas.')
 end
 
 if nargin == 2
     MAX = max( max(A(:)), max(B(:)) );
 end

 MSE = mean2((A-B).*(A-B));
 
 psnr = 10*log10(MAX^2/MSE);

end
 