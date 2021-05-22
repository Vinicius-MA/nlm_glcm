function [ LBP ] = my_lbp( I, P, R )
% Computes the LBP in a P-long neighborhood, and radius R
% Typically, P = 8, R = 1
%
% LBP(c,P,R) = sum(s(gp - gc)*2^p) as p from 0 to P-1
%   c : index of central pixel
%   P : total of neighbors considered
%   R : maximum distance radius from neighbooring and central pixels
%   gc : gray level for the central pixel  - I(c)
%   gp : gray level for the pth neighboring pixel  - I(p)
%   s(x): activation function
%           s(x) = 1, if x>=0; 0 otherwise

    % Get Input Image dimensions
    [M, N] = size(I);

    % Memory for output
    LBP = zeros(M - 2*R, N - 2*R);

    % I(ic, jc) ==> central pixel
    % I(a, b) ==> neighboring pixel
    for ic = 1+R :1: M-R
        for jc = 1+R :1: N-R

            for p = 0:P-1

                %   transforming value of p in copordinates to find the 
                % corespondent neighboor
                if p < P/4                  % top neighborhood                
                    a = ic - R;
                    b = jc - R + p;
                else if p < (2*P)/4         % right neighborhood
                        a = ic - R + ( p - floor(P/4) );
                        b = jc  + R;
                    else if p < (3*P)/4     % bottom nieghborhood
                            a = ic + R;
                            b = jc + R - ( p - floor((2*P)/4) );
                        else if p < P       % left neighborhood
                                a = ic  + R - ( p - floor((3*P)/4) );
                                b = jc - R;
                            end                    
                        end                
                    end            
                end
                
                x1 = I(a,b);
                x2 = I(ic, jc);
                dif = int16(I(a,b)) - int16(I(ic, jc));
                
                step = s( dif );
                value =  step * ( 2^p );

                LBP(ic-R, jc-R) = LBP(ic-R,jc-R) + value;

            end
        end
    end
    
    LBP = mat2gray(LBP, [0, 2^(P-1)]);
    
end

function [y] = s(x)
    
    if x >= 0
        y = 1;
    else
        y = 0;
    end
end
