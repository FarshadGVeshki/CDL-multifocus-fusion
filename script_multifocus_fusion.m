%% multifocus image fusion using coupled dictionary learning
clear
clc
load('Coupled_Dicts.mat')

%% parameters and initialization
p = 8; % patch size
ss = 1; % sliding step
Eps = p^2*1e-4; %approximation threshold
k = 5; % maximum sparsity
param.L = k;
param.eps = Eps;
D = [Df;Db]; D = D./sqrt(sum(D.^2));
%% input image
im1 = double(imread('A.jpg'))/255;
im2 = double(imread('B.jpg'))/255;

%% joint sparse approximation using coupled dictionaries
tstart = tic;
X1 = mexExtractPatches(mean(im1,3),p,ss); X1 = X1 - mean(X1);
X2 = mexExtractPatches(mean(im2,3),p,ss); X2 = X2 - mean(X2);

Xj1 = [X1;X2];
Xj2 = [X2;X1];
% common sparse representation
A1 = mexOMP(Xj1,D,param);
A2 = mexOMP(Xj2,D,param);

%% Fusion

% approximation errors
e1 = sum( (D*A1 - Xj1).^2 ); 
e2 = sum( (D*A2 - Xj2).^2 );

Mv = ones(p^2,1)*double(e1<e2); % vectorized mask

Mask = mexCombinePatches(Mv,zeros(size(im1,1),size(im1,2)),p,0,ss);
Mask(Mask>0.5)=1;
Mask(Mask<=0.5)=0;
ker = 12;
Mask = conv2(Mask,ones(ker)/ker^2,'same');
Mask(Mask>.5)=1;
Mask(Mask<=.5)=0;


imF = Mask.*im1 + (1-Mask).*im2;
toc(tstart)
figure(1)
imshow([imF im1 im2],[])





