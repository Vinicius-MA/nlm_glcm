function [ F ] = im2patch(f, hw)
% im2patch: converte a imagem em uma matriz de patches
% Inputs                                           
%   f	: imagem de entrada                              
%   hw	: meia largura do patch                        
% Outputs                                          
%   F   : matriz de patches                             

% dimensão de entrada
[m n] = size(f);
% largura do patch
w = 2*hw + 1;
% replicar bordas
ff = padarray(f, [hw hw], 'symmetric');
% matriz de patch
F = zeros(m, n, w, w);
patch = @(i,j) ff(i:i+w-1,j:j+w-1);

for i = 1:m
    for j = 1:n
        F(i,j,:,:) = patch(i,j);
    end
end
 y = F(1:5,1:5,:,:)
end
