function  imF = func_multifocus_fusion(im1,im2,Df,Db)
% Multifocus image fusion using coupled dictionary learning
%
% Inputs: multifocus images(im1,im2); coupled multifocus dictionaries(focusd:Df, out of focus:Db)
% Outout: fused image (imF)
%
% reference: F. G. Veshki, N. Ouzir and S. A. Vorobyov, "Image Fusion using Joint Sparse Representations and 
% Coupled Dictionary Learning," ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal
% Processing (ICASSP), Barcelona, Spain, 2020, pp. 8344-8348, doi: 10.1109/ICASSP40776.2020.9054097.

%%% parameters and initialization
D = [Df;Db]; D = D./sqrt(sum(D.^2));
p = sqrt(size(Df,1)); % patch size
ss = 1; % sliding step
Eps = p^2*1e-4; %approximation threshold
k = 5; % maximum sparsity
param.L = k;
param.eps = Eps;

%%% joint sparse approximation
X1 = mexExtractPatches(mean(im1,3),p,ss); X1 = X1 - mean(X1);
X2 = mexExtractPatches(mean(im2,3),p,ss); X2 = X2 - mean(X2);

Xj1 = [X1;X2];
Xj2 = [X2;X1];

% common sparse representation
A1 = mexOMP(Xj1,D,param);
A2 = mexOMP(Xj2,D,param);

%%% Fusion
% approximation errors
e1 = sum( (D*A1 - Xj1).^2 ); 
e2 = sum( (D*A2 - Xj2).^2 );

Mv = ones(p^2,1)*double(e1<e2); % vectorized mask
Mask = mexCombinePatches(Mv,zeros(size(im1,1),size(im1,2)),p,0,ss);
Mask(Mask>0.5)=1;
Mask(Mask<=0.5)=0;
% refining the mask
ker = 12;
Mask = conv2(Mask,ones(ker)/ker^2,'same');
Mask(Mask>.5)=1;
Mask(Mask<=.5)=0;

% masking
imF = Mask.*im1 + (1-Mask).*im2;

end