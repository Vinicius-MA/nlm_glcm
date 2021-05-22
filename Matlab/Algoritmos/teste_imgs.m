clear all;
close all;
clc;

screen_size = get(0,'Screensize');
w = screen_size(3);
h = screen_size(4);
fw = round(w*0.3);
fh = round(h*0.4);
fig_opt = [(w-fw)/2 (h-fh)/2 fw fh];

cmap = [0.7 0.7 0.7];

% IMG
tam = 41;
img_orig = 50*ones(tam);
t = -floor(tam/2):floor(tam/2);
d = repmat(t, tam, 1);
d = sqrt(d.^2+(d').^2);
img_orig(d < floor(tam/4)) = 200;

p = 0.1;
img1 = poissrnd(img_orig/p)*p;
p = 2;
img2 = poissrnd(img_orig/p)*p;
p = 15;
img3 = poissrnd(img_orig/p)*p;
img3 (img3 > 255) = 255;
lim = [min(img3(:)) max(img3(:))];
fsize = 18;

lr = [1 0.5 0.5];
dr = [1 0.1 0.1];
lb = [0.5 0.5 1];
db = [0.1 0.1 1];
lg = [0.5 1 0.5];
dg = [0.1 0.7 0.1];
axc = [0 0 0];

%%
f0 = figure('Position',fig_opt, 'color', [1 1 1]);
    colormap(cmap);
    surf(img1, 'linesmoothing', 'on');
    zlim(lim)
    grid off;
    set(gca, 'View', [-10 45]);
    set(gca, 'box', 'off', 'Xcolor', 'w', 'Ycolor', 'w', ...
        'Zcolor', 'w', 'XTick', [], 'YTick', [], 'ZTick', []);
    save_plot(gca, gcf, 'rel_dose_0', 'png')

%%
f1 = figure('Position',fig_opt, 'color', [1 1 1]);
    colormap(cmap);
    surf(img2, 'linesmoothing', 'on');
    zlim(lim)
    grid off;
    axis off;
    set(gca, 'View', [-10 45]);
    set(gca, 'box', 'off', 'Xcolor', 'w', 'Ycolor', 'w', ...
        'Zcolor', 'w', 'XTick', [], 'YTick', [], 'ZTick', []);
    save_plot(gca, gcf, 'rel_dose_1', 'png')
    
%%
f2 = figure('Position',fig_opt, 'color', [1 1 1]);
    colormap(cmap);
    surf(img3, 'linesmoothing', 'on');
    zlim(lim)
    grid off;
    axis off;
    set(gca, 'View', [-10 45]);
    set(gca, 'box', 'off', 'Xcolor', 'w', 'Ycolor', 'w', ...
        'Zcolor', 'w', 'XTick', [], 'YTick', [], 'ZTick', []);
    save_plot(gca, gcf, 'rel_dose_2', 'png')
