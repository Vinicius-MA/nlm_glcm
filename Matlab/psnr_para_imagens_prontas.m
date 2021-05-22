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

samples = 10;                         %   amostras por imagem carregada
filename = {'HW_C001_120','HW_C002_120','HW_C011_120','HW_C012_120'};   %   nome das imagens utilizadas
method = {'Ruidosa','DDID','BM3D','NLDD','DA3D'};
extension = '.jpg';
sigma = [1, 5, 10, 25, 50, 100];    %   sigmas a serem testados

psnr_result = double(zeros(samples+1,5));

rownames = '';
for i=1:samples
    rownames = [rownames,{num2str(i,'%02d')}];        
    if i == samples
        rownames = [rownames,{'Media'}];
    end
end

for k=1:size(filename,2)
    for i=1:size(sigma,2)
        for j=1:samples            
            for x=1:size(method,2)                
                Y = imread([base_folder,filename{k},extension]);
                Y = imresize(Y(:,:,1),0.75);
                
                Yr = imread([save_folder,filename{k},'_sigma_',num2str(sigma(i),'%03d'),'_Amostra_',num2str(j,'%02d'),'_',method{x},extension]);                
                psnr_result(j,x) = psnr(double(Y),double(Yr));
                
            end                        
        end
        
        %   Salva valores medios na ultima linha
        psnr_result(samples+1,:) = mean(psnr_result(1:samples,:));
        
        T = table(psnr_result(:,1),psnr_result(:,2),psnr_result(:,3), psnr_result(:,4),psnr_result(:,5),...
            'VariableNames',method,'RowNames',rownames);

        writetable(T,[table_folder,filename{k},'.xlsx'],'Sheet',['sigma_',num2str(sigma(i),'%03d')],...
            'Range','A1','WriteRowNames',true);
    end
end