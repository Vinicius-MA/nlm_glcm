clear all
close all
clc

base_folder = fullfile('C:','Users','vinic','Documents','Gradua��o','TCC','Banco de Imagens','Hardware original',filesep);
save_folder = fullfile('C:','Users','vinic','Documents','Gradua��o','TCC','Imagens Obtidas',filesep);
table_folder = fullfile('C:','Users','vinic','Documents','Gradua��o','TCC','Resultados',filesep);

filename = {'HW_C001_120'};
extension = '.jpg';

Im = imread([base_folder,filename{1},extension]);
Im = rgb2gray(Im);

lbp_result = lbp(Im,2,16);
figure,
plot(1:size(lbp_result,2),lbp_result), title('lbp function');
    grid on;
    xlim([1, size(lbp_result,2)]);