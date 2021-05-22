function xt = BFilter(x, y, w, sigma_d, sigma_r)
    [dx, dy] = meshgrid(-w:w);
    h = exp(- (dx.^2 + dy.^2) / (2 * sigma_d^2));
    xp = padarray(x, [w w], 'symmetric');
    yp = padarray(y, [w w], 'symmetric');
    xt = zeros(size(x));
    for p = 1:numel(x), [i, j] = ind2sub(size(x), p);
        % Spatial Domain: Bilateral Filter
        g = xp(i:i+2*w, j:j+2*w);
        s = yp(i:i+2*w, j:j+2*w);
        d = g - g(1+w, 1+w);
        k = exp(- d.^2 ./ (sigma_r)) .* h;
        xt(p) = sum(sum(s .* k)) / sum(k(:));
    end
end