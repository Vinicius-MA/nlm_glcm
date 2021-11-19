clear all
close all
clc

base_folder = 'Banco de Imagens\';
input_folder = 'NovoNoisy\';
save_folder = 'NovoObtidas\';
table_folder = 'NovoResultados\';

samples = 10;                         %   amostras por imagem carregada
im_code = [1, 2, 4, 9, 11, 12, 13, 15, 16, 17, 19, 24];
extension = '.jpg';
sigma = [10, 25, 50];    %   sigmas a serem testados

filename = cell(1, size(im_code,2) );  %   nome das imagens utilizadas
for k = 1: size(im_code,2)
   filename{k} = sprintf('HW_C%03d_120', im_code(k) ); 
end

rownames = '';
for i=1:samples
    rownames = [rownames,{num2str(i,'%02d')}];        
    if i == samples
        rownames = [rownames,{'Media'}];
    end
end

for k=1:size(filename,2)

    % Y in [0, 1]
    Y = mat2gray( rgb2gray( imread( [base_folder,filename{k},extension]) ), [0, 255] );

    for i=1:size(sigma,2)
        psnr_result = noiseSampling( Y,sigma(i),samples,input_folder, save_folder, filename{k} ) ;

        T = table(psnr_result(:,1),psnr_result(:,2),psnr_result(:,3), psnr_result(:,4), psnr_result(:,5), psnr_result(:,6),...
            'VariableNames',{'Noisy','DDID','BM3D','NLDD', 'DA3D', 'NLM'},'RowNames',rownames);

        writetable(T,[table_folder,filename{k},'_matlab.xlsx'],'Sheet',['sigma_',num2str(sigma(i),'%03d')],...
            'Range','A1','WriteRowNames',true);
    end
end
