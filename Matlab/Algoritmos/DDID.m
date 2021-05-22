function x = DDID(y, sigma2)
    x = step(y, y, sigma2, 15, 7, 100, 4.0);
    x = step(x, y, sigma2, 15, 7, 8.7, 0.4);
    x = step(x, y, sigma2, 15, 7, 0.7, 0.8);
end

function xt = step(x, y, sigma2, r, sigma_s, gamma_r, gamma_f)
    [dx, dy] = meshgrid(-r:r);
    h = exp(- (dx.^2 + dy.^2) / (2 * sigma_s^2));
    xp = padarray(x, [r r], 'symmetric');
    yp = padarray(y, [r r], 'symmetric');
    xt = zeros(size(x));
    parfor p = 1:numel(x), [i, j] = ind2sub(size(x), p);
        % Spatial Domain: Bilateral Filter
        g = xp(i:i+2*r, j:j+2*r);
        y = yp(i:i+2*r, j:j+2*r);
        d = g - g(1+r, 1+r);
        k = exp(- d.^2 ./ (gamma_r * sigma2)) .* h; % Eq. 4
        gt = sum(sum(g .* k)) / sum(k(:)); % Eq. 2
        st = sum(sum(y .* k)) / sum(k(:)); % Eq. 3
        
        % Fourier Domain: Wavelet Shrinkage
        V = sigma2 .* sum(k(:).^2); % Eq. 5
        G = fft2(ifftshift((g - gt) .* k)); % Eq. 6
        S = fft2(ifftshift((y - st) .* k)); % Eq. 7
        K = exp(- gamma_f * V ./ (G .* conj(G))); % Eq. 9
        St = sum(sum(S .* K)) / numel(K); % Eq. 8
        xt(p) = st + real(St); % Eq. 1
    end
end