%% learning coupled dictionaries from multifocus training data
clear
clc
addpath('couple_dictionary_learning_codes')
addpath('training_data')

%% parameters
P = 30000; % max number of training patches
p = 8; % patch size
ss = 1; % sliding step
Eps = p^2*1e-4; %approximation threshold
w = 0.5; % tuning parameter omega
param.omega = w; 
opts.eps = Eps; 
opts.k = 5; % maximim number of nonzeros in sparse vectors
opts.K = 128; % number of atoms in dictionaries
opts.Nit = 20; % number of CDL iterations
opts.remMean = true; % removing mean from the samples
opts.DCatom = true; % first atom is DC atom
opts.print = true; % printing the results

%% prepairing training data
Xf = []; % infocus patches
Xb = []; % out of focus

for i = 1:5
   temp = double(imread(['p' num2str(i) 'f.png']))/255;
   Xf = [Xf mexExtractPatches(temp,p,ss)];

   temp = double(imread(['p' num2str(i) 'b.png']))/255;
   Xb = [Xb mexExtractPatches(temp,p,ss)];
end
V = var([Xf;Xb]); % make sure patches with low variance are removed from traing data
Xf(:,V<Eps) = [];
Xb(:,V<Eps) = [];

Inds = randperm(size(Xf,2));
Xf = Xf(:,Inds(1:P)); Xf = Xf - mean(Xf);
Xb = Xb(:,Inds(1:P)); Xb = Xb - mean(Xb);

%% learning coupled dictionaries

[Df, Db] = CDL(Xf,Xb,opts);

ID1 = displayPatches(Df);
ID2 = displayPatches(Db);

figure(2)
subplot 121
imshow(ID1)
title('Df')
subplot 122
imshow(ID2)
title('Db')

% save('Coupled_Dicts','Df','Db')

    
