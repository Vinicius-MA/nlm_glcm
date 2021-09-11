function [ u ] = nlm_lbp_andre(v, d, s, h)

%% Tentativa de implementação de NLM-LBP com algortimo do André
% ------------------------------------------------------------------------
% Entrada                                           
%   v   : imagem ruidosa
%   d   : distancia da janela de busca ( Si[d, d] )
%   s   : distancia do patch ( Li[l, l] )
%   h   : parametro de filtragem (para NLM)
% ------------------------------------------------------------------------


%% tamanho da imagem de entrada
[m, n] = size(v);
% imagem de saida
u = zeros([m n]);

%% NLM
h = h*h;
% matriz de soma dos pesos
w_nlm = zeros([m n]);
% matriz dos pesos maximos
z_nlm = zeros([m n]);
% matriz de patches
P_nlm = im2patch(v, s);

% função de distância
f_nlm = @(x1, x2) (x1 - x2).^2;

% replicar bordas das matrizes da imagem e de patch
v1_nlm = padarray(v, [d d], 'symmetric');
P1_nlm = padarray(P_nlm, [d d 0 0], 'symmetric');

%% LBP
% matriz de soma dos pesos
w_lbp = zeros([m n]);
% matriz dos pesos maximos
z_lbp = zeros([m n]);
% matriz de patches
P_lbp = im2patch(v, s);

% replicar bordas das matrizes da imagem e de patch
v1_lbp = padarray(v, [d d], 'symmetric');
P1_lbp = padarray(P_nlm, [d d 0 0], 'symmetric');

%%
% offset das bordas
x = (1:m) + d;	y = (1:n) + d;

% deslocamentos da janela de busca 
image_nlm = @(dx,dy) v1_nlm(x+dx,y+dy);
patch_lnm = @(dx,dy) P1_nlm(x+dx,y+dy,:,:);

patch_lbp = @(dx, dy) P1_lbp(x+dx, y+dy);


for dx = -d:d
for dy = -d:d
    
    %% LOOP PRINCIPAL
    tic
    
	% pular o deslocamento dx,dy = (0,0) (não comparar o patch central com ele mesmo)
    if (dx ~= 0 || dy ~= 0)
        
        %% NLM

        % desloca as matrizes
        vxy_nlm = image_nlm(dx,dy);
        Pxy_nlm = patch_lnm(dx,dy);
        % calcula as distancias 
        dxy_nlm = sum(sum( f_nlm(P_nlm,Pxy_nlm), 4), 3);
        % calcula os pesos
        wxy_nlm = exp( -dxy_nlm/h );
        % atualiza matriz com os pesos maximos
        z_nlm = max(z_nlm, wxy_nlm);
        % atualiza a matriz da soma dos pesos
        w_nlm = w_nlm + wxy_nlm;
        
        %% LBP
        
        % desloca as matrizes
        Pxy_lbp = patch_lbp(dx,dy);
        % calcula as distancias 
        dxy_lbp = f_lbp( P_lbp, Pxy_lbp);
        % desvio padrão ( hSi )
        hSi = std( std( dxy_lbp) );        
        % calcula os pesos
        wxy_lbp = exp( -dxy_lbp / hSi );
        % atualiza matriz com os pesos maximos
        z_lbp = max(z_lbp, wxy_lbp);
        % atualiza a matriz da soma dos pesos
        w_lbp = w_lbp + wxy_lbp;
        
        %% Resultado
        
        % atualiza a imagem de saida com o peso dos patches do deslocamento atual
        u = u + (wxy_nlm .* wxy_lbp ) .* vxy_nlm;
        
        t = toc;    
        fprintf('pixel (%d, %d) of (%d, %d)  processed in %.03f\r\n', dx, dy, 2*d+1, 2*d+1, t);
        
        
    end
    
end
end

%% NLM

% atribui o peso maximo para o pixel central
u = u + v.*z_nlm;
% incrementa a matriz de soma dos pesos com os pesos do pixel central
w_nlm = w_nlm + z_nlm;
% normaliza os valores obtidos com a soma dos pesos
u = u./w_nlm;

%% LBP
% atribui o peso maximo para o pixel central
u = u + v.*z_lbp;
% incrementa a matriz de soma dos pesos com os pesos do pixel central
w_lbp = w_lbp + z_lbp;
% normaliza os valores obtidos com a soma dos pesos
u = u./w_lbp;

end

function [d] = f_lbp(P1, P2)

m = size(P1, 1) ;
n = size( P1, 2 );

d = zeros([m, n]);

for i=1:m
    for j = 1:n
        
        H1 = imhist( P1(i, j, :, :) );
        H2 = imhist( P2(i, j, :, :) );
        
        d(i, j) = sum( (H1 - H2).^2 ./ (H1 + H2+0.00001) );
    
    end
end

end
