function [ u ] = nlmfilter(v, d, s, f, h)
% Implementação do filtro Nonlocal-Means 
% -------------------------------------------------
% Entrada                                           
%   v   : imagem ruidosa
%   d   : distancia da janela de busca
%   s   : distancia do patch
%   f   : função de similaridade
%   h   : parametro de filtragem
% Saida                                          
%   u   : imagem filtrada

h = h*h;
% tamanho da imagem de entrada
[m, n] = size(v);
% imagem de saida
u = zeros([m n]);
% matriz de soma dos pesos
w = zeros([m n]);
% matriz dos pesos maximos
z = zeros([m n]);
% matriz de patches
P = im2patch(v, s);

% replicar bordas das matrizes da imagem e de patch
v1 = padarray(v, [d d], 'symmetric');
P1 = padarray(P, [d d 0 0], 'symmetric');

% offset das bordas
x = (1:m) + d;	y = (1:n) + d;
% deslocamentos da janela de busca 
image = @(dx,dy) v1(x+dx,y+dy);
patch = @(dx,dy) P1(x+dx,y+dy,:,:);

for dx = -d:d
for dy = -d:d
    
	% pular o deslocamento dx,dy = (0,0) (não comparar o patch central com ele mesmo)
    if (dx ~= 0 || dy ~= 0) 

        % desloca as matrizes
        vxy = image(dx,dy);
        Pxy = patch(dx,dy);
        % calcula as distancias 
        dxy = sum(sum( f(P,Pxy), 4), 3);
        % calcula os pesos
        wxy = exp( -dxy/h );
        % atualiza matriz com os pesos maximos
        z = max(z, wxy);
        % atualiza a imagem de saida com o peso dos patches do deslocamento atual
        u = u + wxy.*vxy;
        % atualiza a matriz da soma dos pesos
        w = w + wxy;
        
    end
    
end
end

% atribui o peso maximo para o pixel central
u = u + v.*z;
% incrementa a matriz de soma dos pesos com os pesos do pixel central
w = w + z;
% normaliza os valores obtidos com a soma dos pesos
u = u./w;

end
