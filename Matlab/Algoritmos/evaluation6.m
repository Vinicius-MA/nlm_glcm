function[SSIM_MAP,results] = evaluation6(A,B,print,L) 

    % Convencao:
    %   A: Imagem degradada
    %   B: Ground truth

    % Maior precisao nos resultados
    format long;
    
    if (~exist('print', 'var'))
       print = 0; 
    end
    
    if (~exist('L', 'var'))
        L = max(max(A(:), max(B(:))));
%         if print,
%             fprintf('\nL automatico: %d\n', L);
%         end
    end
    
    [M,N] = size(A);

    % SNR
    SNR = sum(sum(A.^2)) / sum(sum((A-B).^2));
    SNRdb = 10*log10(SNR);
    
    % NMSE
    MSE = sum(sum((A-B).^2))/(M*N);
    MA = sum(sum(A))/(M*N);
    MB = sum(sum(B))/(M*N);
    NMSE = MSE/(MA*MB);
    
    % PSNR
    PSNR = (L^2)/MSE;
    %disp(log10(PSNR));
    PSNRdb = 10*log10(PSNR);
    
    % MSSIM
    K = [0.01 0.03]; 
    %windows = fspecial('gaussian', 11, 1.5);
    windows = ones(8);
    [MSSIM, SSIM_MAP] = ssim_index(A, B, K, windows, L);

    % WNPSSIM
    WNPSSIM = WNPSSIM_index(A, B);
    
    % Sharpness
    sharp = estimate_sharpness2(A);

    if print,
        fprintf('\t SNR: %f\n', SNR);
        fprintf('\t SNR (dB): %f\n', SNRdb);
        fprintf('\t NMSE: %f\n', NMSE);
        fprintf('\t PSNR: %f\n', PSNR);
        fprintf('\t PSNR (dB): %f\n', PSNRdb);
        fprintf('\t MSSIM: %f\n', MSSIM);
        fprintf('\t WNPSSIM: %f\n', WNPSSIM);
        fprintf('\t Sharpness (dB): %f\n', sharp);
    end
        
    results = zeros(1,8);
    results(1) = SNR;
    results(2) = SNRdb;
    results(3) = NMSE;
    results(4) = PSNR;
    results(5) = PSNRdb;
    results(6) = MSSIM;
    results(7) = WNPSSIM;
    results(8) = sharp;
               
end
