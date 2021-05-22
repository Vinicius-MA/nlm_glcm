clear all
close all
clc

base_folder = (fullfile('C:','Users','vinic','Google Drive','USP','Disciplinas',...
    '8º Semestre','SEL0442 - Projeto de Formatura I','Banco de Imagens','Hardware original',filesep));
save_folder = fullfile('C:','Users','vinic','Google Drive','USP','Disciplinas','8º Semestre',...
    'SEL0442 - Projeto de Formatura I','MATLAB','Imagens Obtidas',filesep);
table_folder = fullfile('C:','Users','vinic','Google Drive','USP','Disciplinas','8º Semestre',...
    'SEL0442 - Projeto de Formatura I','MATLAB','Resultados',filesep);
original_subtitle = '_Imagem_Original';

samples = 10;               %   amostras por imagem carregada
filename = 'HW_C001_120'; %   nome das imagens utilizadas
extension = '.jpg';
sigma = 25;               %   sigmas a serem testados

    Y = imread([base_folder,filename,extension]);
    Y = (imresize(Y(:,:,1),0.75));
    
    v = (sigma/(2.^8-1)).^2;
    
    %   Obter Imagem ruidosa
    Yn = imnoise(Y,'gaussian',0,v);
    
    %   Filtro NLDD
    Ynldd = uint8(NLDD(double(Yn), sigma, 1));

    %   DA3D + NL-Bayes
    Ybayes = NlBayesDenoiser(double(Yn), sigma, 1);
    Yda3d = uint8(DDIDstep(Ybayes,double(Yn),sigma.^2, 15, 7, 0.7, 0.8));
    
    figure, imshow(Yn), title('Ruidosa');
    figure, imshow(Ynldd), title('NLDD');
    figure, imshow(Yda3d), title('DA3D');
    
    imwrite(Yda3d,'teste.jpg');