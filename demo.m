addpath('tensor_toolbox')
Ntr = 100;
p = 2;
I1 = 3;
I2 = 3;

Xtr = randn(Ntr,p);
B = zeros(I1,I2,p);
B(:,:,1) = [4 1 0;1 0.1 0; 1 0 1];
B(:,:,2) = [1 2 0;1 3 0;1 0 0.2];

n1 = 100
n2 = 50
N = size(Xtr,1);
Ytr = zeros(n1,n2,N);
I1 = 3;
I2 = 3;
cU{1} = [sin((1:n1)/n1*pi)' sin((1:n1)/n1*pi*2)' sin((1:n1)/n1*pi*3)']/sqrt(100);
cU{2} =[sin((1:n2)/n2*pi)' sin((1:n2)/n2*pi*2)' sin((1:n2)/n2*pi*3)']/sqrt(100);
%B = double(randn(I1,I2,p)<0.1);
%B(B~=0) =;
sigma_mag = 0.001; 
A = double(ttm(tensor(B),{cU{1},cU{2}},[1,2]));
Ytr = ttm(tensor(B),{cU{1},cU{2},Xtr});
Ytr = Ytr +  sigma_mag*normrnd(0,1,size(Ytr));

%% Showing the samples
isample = 3
imagesc(double(Ytr(:,:,isample)))
title([num2str(Xtr(isample,1)),' ',num2str(Xtr(isample,2))])

%% Method 1: Tensor Regression with basis


options.I1 = 3;
options.I2 = 3;
options.lambda2 =[];

theta1 = 0.1;
k1 = 10;
k2 = 10
% PCA
isiid = 1
[ Sigma,S,invS] = createSigma2( n1,n2,theta1,0,isiid);


B = cell(2,1);
B{1} = bsplineBasis(n1,k1);
B{2} = bsplineBasis(n2,k2);


lambda1 = 1;
[ alpha,Yhattr,Vhat,lambdanew,gcvscore] = tensorreg(Xtr,Ytr,[],Xte,B,lambda1,Sigma);

subplot(2,2,1)
imagesc(alpha(:,:,1))
title('estimated 1')
subplot(2,2,2)
imagesc(alpha(:,:,2))
title('estimated 1')

subplot(2,2,3)
imagesc(A(:,:,1))
title('true 1')


subplot(2,2,4)
imagesc(A(:,:,2))
title('true 2')


%% Method 2: Tucker Regression
iidmatrix = 1;
maxiters = 1000;
[ iidmatrix,S,invS] = createSigma( n1,n2,theta,0,1);
[core,alpha,Uinit] = tucker_onesetp(Xtr,Ytr,R,maxiters,iidmatrix);
subplot(2,2,1)
imagesc(alpha(:,:,1))
title('estimated 1')
subplot(2,2,2)
imagesc(alpha(:,:,2))
title('estimated 1')

subplot(2,2,3)
imagesc(A(:,:,1))
title('true 1')


subplot(2,2,4)
imagesc(A(:,:,2))
title('true 2')
