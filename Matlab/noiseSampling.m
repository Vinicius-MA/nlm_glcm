%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               COMPARAÇÃO DE MÉTODOS BM3D, DDID, NLDD          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Ruído utilizado: Gaussiano, média = 0, variância = sigma^2
function [ psnr_result ] = noiseSampling( Y, sigma, samples, input_folder, save_folder, filename)

    
    n_methods = 6;

    extension =         '.jpg';
    noisy_subtitle =    '_noisy';
    ddid_subtitle =     '_ddid';
    bm3d_subtitle =     '_bm3d';
    nldd_subtitle =     '_nldd';
    da3d_subtitle =     '_da3d';
    nlm_subtitle =      '_nlm_matlab';

    psnr_result = double(zeros(samples+1, n_methods));
    
    % sigma in [0, 255] ; v in [0, 1]
    v = (sigma/(2.^8-1)).^2;
    
    for i =1:samples
    
        upper_title = [filename,'_sigma',num2str(sigma,'%03d'),'_',num2str(i,'%02d')];        
        
        fprintf( "%s\r\n", upper_title ) ;

        %   Noisy -- [0, 1]
        if isfile( [ input_folder, upper_title, noisy_subtitle, extension ] )
            Yn = mat2gray( imread( [ input_folder, upper_title, noisy_subtitle, extension ] ), [0, 255] );
        else
            Yn = imnoise(Y, 'gaussian', 0, v);
        end
        
        fprintf("\tddid\r\n");

        %   DDID
        if isfile( [ save_folder, upper_title, ddid_subtitle, extension ] )
            Yddid = mat2gray( imread( [ save_folder, upper_title, ddid_subtitle, extension ] ), [0, 255] );
        else
            Yddid = DDID( Yn, v);
        end
            

        %   BM3D
        if isfile( [ save_folder, upper_title, bm3d_subtitle, extension ] )
            Ybm3d = mat2gray( imread( [ save_folder, upper_title, bm3d_subtitle, extension ] ), [0, 255] );
        else
            % must provide sigma even with Yn in [0,1]
            [~, Ybm3d] = BM3D( 1, Yn, sigma );
        end
            
            
        %   NLDD
        if isfile( [ save_folder, upper_title, nldd_subtitle, extension ] )
            Ynldd = mat2gray( imread( [ save_folder, upper_title, nldd_subtitle, extension ] ), [0, 255] );
        else
            %   passing standar deviation as argument
            Ynldd = NLDD(Yn, v^0.5, 1);
        end
            
            
        %   DA3D + NL-Bayes
        if isfile( [ save_folder, upper_title, da3d_subtitle, extension ] )
            Yda3d = mat2gray( imread( [ save_folder, upper_title, da3d_subtitle, extension ] ), [0, 255] );
        else
            %   passing standar deviation as argument
            Ybayes = NlBayesDenoiser(Yn, v^0.5, 1);
            Yda3d = DDIDstep( Ybayes, Yn, v, 15, 7, 0.7, 0.8);   %   os últimos 4 parâmetros, não sei o que são 
        end
        
        %   Filtro NLM
        if isfile( [ save_folder, upper_title, nlm_subtitle, extension ] )
            Ynlm = mat2gray( imread( [ save_folder, upper_title, nlm_subtitle, extension ] ), [0, 255] );
        else
            Ynlm = nlm( Y, Yn, 10, 3, v);
        end
            

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
            imwrite(Yn,[save_folder,upper_title,noisy_subtitle,extension]);

            %   Imagem DDID
            imwrite(Yddid,[save_folder,upper_title,ddid_subtitle,extension]);

            %   Imagem BM3D 
            imwrite(Ybm3d,[save_folder,upper_title,bm3d_subtitle,extension]);
            
            %   Imagem NLDD
            imwrite(Ynldd,[save_folder,upper_title,nldd_subtitle,extension]);
            
            %   Imagem DA3D + NL-Bayes
            imwrite(Yda3d,[save_folder,upper_title,da3d_subtitle,extension]);
            
            %   Imagem NLM
            imwrite(Ynlm, [save_folder, upper_title, nlm_subtitle, extension]);
    end
end