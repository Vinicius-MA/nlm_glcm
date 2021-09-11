function [D] = chi_sqr_diss( A, B, patch_r )

[m, n] = size(A);

D = zeros(m,n);

A_pad = im2patch( A, patch_r);
B_pad = im2patch( B, patch_r);

for i = 1:K
    
    Ai = A_pad( patch_r + (1:m) + Y(i), patch_r + (1:n) + X(i) );
    Bi = B_pad( patch_r + (1:m) + Y(i), patch_r + (1:n) + X(i) );
    
    hist_Ai = imhist(Ai);
    hist_Bi = imhist(Bi);
    
    for N = 1:size(hist_Ai)
        
        if ( hist_Ai(N) == 0 ) && ( hist_Bi(N) ==0 )
        continue;
        end

        D(i) = D(i) + ( hist_Ai(N) - hist_Bi(N) ).^2 ./ ( hist_Ai(N) + hist_Bi(N) ) ;
    end
    
end

end