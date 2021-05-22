clear all
close all
clc

base_foldername = fullfile('C:','Users','vinic','Documents','Graduação','TCC','Banco de Imagens','Hardware original',filesep);
save_foldername = fullfile('C:','Users','vinic','Documents','Graduação','TCC','Imagens Obtidas',filesep);

filename = 'HW_C011_120';
extension = '.jpg';
original_subtitle = '_0_Imagem_Original';

noisy_subtitle = '_1_Ruidosa';
ddid_subtitle = '_2_DDID';
bm3d_subtitle = '_3_BM3D';

samples = 10;
sigma = 25;

y = imread([base_foldername,filename,extension]);
y = (imresize(y(:,:,1),0.5));

%   Imagem Original
    h = figure; imshow(y);    
        imwrite(y,[save_foldername,filename,...
            original_subtitle,extension]);
        close(h);

psnr_result = double(zeros(samples+1,4));
y_media = double(zeros(size(y)));
rownames = '';

for i =1:samples
    
    upper_title = [filename,'_Amostra_',num2str(i,'%02d')];
    
    
        rownames = [rownames,{num2str(i,'%02d')}];        
        if i == 10
            rownames = [rownames,{'Media'}];
        end
    
    %   Adicionando Ruído
        y_n = imnoise(y,'gaussian',0,2/255);
        
    %   Media das imagens
        y_media = y_media + ( double(y_n)/samples );

    %   Filtro DDID
        y_ddid = uint8(DDID(double(y_n),sigma.^2));

    %   Filtro BM3D
        [~, y_bm3d] = BM3D(double(y),double(y_n),sigma,'np',0);
        y_bm3d = im2uint8(y_bm3d);

    %   PSNRs (noisy, DDID, BM3D)
        psnr_result(i,:) = [psnr(double(y),double(y_n)),psnr(double(y),...
            double(y_ddid)),psnr(double(y),double(y_bm3d)),psnr(double(y),y_media)];
        
        %   última linha, valores médios
        psnr_result(samples+1,:) = psnr_result(samples+1,:)+( psnr_result(i,:)/samples );

    %   IMAGENS
        
    %   Imagem Ruidosa 
        h = figure; imshow(y_n);     
            imwrite(y_n,[save_foldername,upper_title,...
                noisy_subtitle,extension]);
            
            close(h);
     
    %   Imagem DDID
        h = figure; imshow(y_ddid);            
            imwrite(y_ddid,[save_foldername,upper_title,...
                ddid_subtitle,extension]);
            
            close(h);
            
    %   Imagem BM3D 
        h = figure; imshow(y_bm3d);        
            imwrite(y_bm3d,[save_foldername,upper_title,...
                bm3d_subtitle,extension]);            
            close(h);
    
end

T = table(psnr_result(:,1),psnr_result(:,2),psnr_result(:,3), psnr_result(:,4),...
    'VariableNames',{'Noisy','DDID','BM3D','Media'},'RowNames',rownames);

writetable(T,'PSNR_results.xlsx','Sheet',filename,'Range','A1','WriteRowNames',true);