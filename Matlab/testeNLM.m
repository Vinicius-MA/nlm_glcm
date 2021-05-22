clear all
close all
clc

base_folder = fullfile('C:','Users','vinic','Documents','Graduação','TCC','Banco de Imagens','Hardware original',filesep);
save_folder = fullfile('C:','Users','vinic','Documents','Graduação','TCC','Imagens Obtidas',filesep);
table_folder = fullfile('C:','Users','vinic','Documents','Graduação','TCC','Resultados',filesep);

filename = {'HW_C001_120'};
extension = '.jpg';

I = imread([base_folder,filename{1},extension]);
I = rgb2gray(I);

sigma = 25;
v = (sigma/(2.^8-1)).^2;

In = imnoise(I,'gaussian',0,v);

%Y = NLmeansfilter(In,10,5,1);
input = In;
t = 21;
f = 7;
h = sigma;

 % Size of the image
 [m n]=size(input);
 
 
 % Memory for the output
 Output=zeros(m,n);

 % Replicate the boundaries of the input image
 input2 = padarray(input,[f f],'symmetric');
 
 % Used kernel
 kernel = make_kernel(f);
 kernel = kernel / sum(sum(kernel));
 
 h=h*h;
 
 for i=1:m
 for j=1:n
                 
         i1 = i+ f;
         j1 = j+ f;
                
         W1= input2(i1-f:i1+f , j1-f:j1+f);
         
         wmax=0; 
         average=0;
         sweight=0;
         
         rmin = max(i1-t,f+1);
         rmax = min(i1+t,m+f);
         smin = max(j1-t,f+1);
         smax = min(j1+t,n+f);
         
         for r=rmin:1:rmax
         for s=smin:1:smax
                                               
                if(r==i1 && s==j1) continue; end;
                                
                W2= input2(r-f:r+f , s-f:s+f);                
                 
                d = sum(sum(kernel.*double(W1-W2).*double(W1-W2)));
                                               
                w=exp(-d/h);                 
                                 
                if w>wmax                
                    wmax=w;                   
                end
                
                sweight = sweight + w;
                average = average + w*input2(r,s);                                  
         end 
         end
             
        average = average + wmax*input2(i1,j1);
        sweight = sweight + wmax;
                   
        if sweight > 0
            output(i,j) = average / sweight;
        else
            output(i,j) = input(i,j);
        end                
 end
 end
 
figure, imshow(In);
figure, imshow(output);
figure, imshow(Output);