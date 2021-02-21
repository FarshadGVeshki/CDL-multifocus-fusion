%% multifocus image fusion using coupled dictionary learning
clear
clc
addpath('couple_dictionary_learning_codes')
load('Coupled_Dicts_8.mat')
% load('Coupled_Dicts_16.mat')
%% input images
im1 = double(imread('A.jpg'))/255;
im2 = double(imread('B.jpg'))/255;

%% Fusion
tic
[imf, Mask] = func_multifocus_fusion(im1,im2,Df,Db);
toc

%%

figure(1)
subplot 131
imshow(im1,[])
title('input 1')
subplot 132
imshow(im2,[])
title('input 2')
subplot 133
imshow(imf,[])
title('fused image')

%% 

% figure(2)
% imshow(Mask,[])
