%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               COMPARAÇÃO DE MÉTODOS BM3D, DDID, NLDD          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Ruído utilizado: Gaussiano, média = 0, variância = sigma^2
function [psnr_result]=noiseSampling(Y,sigma,samples,save_foldername,filename)

    
n_methods = 6;

    extension =         '.jpg';
    noisy_subtitle =    '_Ruidosa';
    ddid_subtitle =     '_DDID';
    bm3d_subtitle =     '_BM3D';
    nldd_subtitle =     '_NLDD';
    da3d_subtitle =     '_DA3D';
    nlm_subtitle =      '_NLM';

    psnr_result = double(zeros(samples+1, n_methods));
    v = (sigma/(2.^8-1)).^2;
    
    for i =1:samples
    
        upper_title = [filename,'_sigma',num2str(sigma,'%03d'),'_sample',num2str(i,'%02d')];        

        %   Obter Imagem ruidosa
            Yn = imnoise(Y, 'gaussian', 0, v);

        %   Filtro DDID
            Yddid = DDID( Yn, v);

        %   Filtro BM3D
            [~, Ybm3d] = BM3D(Y, Yn, sigma,'np',0);
            %Ybm3d = im2uint8(Ybm3d);
            
        %   Filtro NLDD
            Ynldd = NLDD(Yn, v^0.5, 1);
            
        %   DA3D + NL-Bayes
            Ybayes = NlBayesDenoiser(Yn, v^0.5, 1);
            Yda3d = DDIDstep( Ybayes, Yn, v, 15, 7, 0.7, 0.8);   %   os últimos 4 parâmetros, não sei o que é
        
        %   Filtro NLM
            Ynlm = nlm( Y, Yn, 10, 3, 10*v);

        %   PSNRs (noisy, DDID, BM3D)
            % noisy, DDID, BM3D, NLDD, DA3D, NLM
            psnr_result(i,:) = [
                psnr(Y,Yn),...
                psnr(Y,Yddid),...
                psnr(Y,Ybm3d),...
                psnr(Y,Ynldd),...
                psnr(Y,Yda3d),...
                psnr(Y,Ynlm)];

        %   Salva valores medios na ultima linha
            psnr_result(samples+1,:) = psnr_result(samples+1,:)+(psnr_result(i,:)/samples );

         %  SALVAR IMAGENS
            %   Imagem Ruidosa
            imwrite(Yn,[save_foldername,upper_title,noisy_subtitle,extension]);

            %   Imagem DDID
            imwrite(Yddid,[save_foldername,upper_title,ddid_subtitle,extension]);

            %   Imagem BM3D 
            imwrite(Ybm3d,[save_foldername,upper_title,bm3d_subtitle,extension]);
            
            %   Imagem NLDD
            imwrite(Ynldd,[save_foldername,upper_title,nldd_subtitle,extension]);
            
            %   Imagem DA3D + NL-Bayes
            imwrite(Yda3d,[save_foldername,upper_title,da3d_subtitle,extension]);
            
            %   Imagem NLM
            imwrite(Ynlm, [save_foldername, upper_title, nlm_subtitle, extension]);
    end
end