% desabilitando os warnings
warning off all;

clear all
close all
clc;

patch = 'r1';

salvar = 1;
mostrar_imagens = 0;

%doses = {'100%', '25%'};
%comp = {'', '_quarterDose'}; 

doses = {'100%', '50%', '25%'};
comp = {'', '_halfDose', '_quarterDose'};

[~, tam] = size(doses);

% arquivo que irá armazenar os resultados
filename = 'resultados.csv';
delete(filename);

v = (25/(2.^8-1)).^2;

% começa do 50%
for pos = 2: tam
    
    ground_NoNoise = rgb2gray(imread('HW_C001_000.jpg'));
    im = double(imnoise( ground_NoNoise, 'gaussian', 0, v ));
    ground_NoNoise = double(ground_NoNoise); 
        
    max_im = max(im(:)); 
    max_NoNoise = max(ground_NoNoise(:));
    fator = max(mean(im(:)), mean(ground_NoNoise(:))) / min(mean(im(:)), mean(ground_NoNoise(:)));
    ground_NoNoise = ground_NoNoise * fator;
    
    L = max_im;
    fprintf('\n\n--> %s %s (Max: %d)\n', doses{pos}, patch, L);
    fprintf('--> (Antes) Max Im: %d; Max Ground: %d', max(im(:)), max(ground_NoNoise(:)));
    
    fprintf('\n--> Evaluation BEFORE Restoration:\n');
    %[~, r] = evaluation3(ground_NoNoise, im, L);

    % NITRO!
%     sigma = 15;

    % É NECESSÁRIO NORMALIZAR A IMG DE ENTRADA ENTRE [0, 1];
    im = im/max_im;

% v = double(im);
% distancia do pixel central até a borda da janela de busca
d = 3; % largura da janela de busca = 2*d + 1
% distancia do pixel central até a borda do patch
s = 1; % largura do patch = 2*s + 1
% distancia euclidiana
f = @(x1,x2) (x1-x2).^2;
% parametro de filtragem
h = 0.024;

    tic;
    y_est = nlmfilter(im,d,s,f,h);   
    t1 = toc;
    
    im = im*max_im;
    
    fator = max(mean(ground_NoNoise(:)), mean(y_est(:))) / min(mean(ground_NoNoise(:)), mean(y_est(:)));
    y_est = y_est * fator;
    max_y_est = max(y_est(:));
    
    
    % Se houver  a necessidade de salvar a imagem pós-procesamento
    if salvar == 1
        imwrite(uint16(y_est), ...
                sprintf('projMarcelo50perc_%s_NLM_%s.tif', doses{pos}, patch));
    end
    

    L = max(max_NoNoise, max_y_est);


    fprintf('\n--> Evaluation After Restoration: (Sigma: %d)\n', h);
    
    %[~, r] = evaluation3(ground_NoNoise, y_est, L);
    
    % salvando em arquivo
    %dlmwrite(filename, r, '-append', 'precision', '%f', 'delimiter', ';');
    
%     ground_NoNoise = imcomplement(ground_NoNoise);
%     y_est = imcomplement(y_est);
%     im = imcomplement(im);
    
    % Imagens e Gráfico! COMENTAR SE NAO FOR USAR!
    x = [0:1:255];
    y_im = im(70,:);
    y_yest = y_est(70,:);
    y_g = ground_NoNoise(70,:);
    figure, plot(x, y_g, x, y_yest,x,y_im);
    legend('NoNoise', 'Restaurada', 'Noisy');
    
    if mostrar_imagens == 1
        figure, imshow(1-mat2gray(ground_NoNoise), []), title('NoNoise');
        figure, imshow(1-mat2gray (y_est), []), title('Restaurada');
        figure, imshow(1-mat2gray(im), []), title('Noisy');
    end
% title ()
end


% figure, imshow(ground_NoNoise, []);
% figure, imshow(y_est, []);


fprintf('\n--> THE END <--\n');


