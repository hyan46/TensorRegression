function [ A,U,para] = onesteptensoreg( Y,X,R,maxiters,isoptimize,isoptimizesigma,isiid,theta)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here


n1 = size(Y,1);
n2 = size(Y,2);
m = n1*n2;
K = 2;



[ iidmatrix,S,invS] = createSigma( n1,n2,theta,0,1);
[core,A,Uinit] = tucker_onesetp(X,Y,R,maxiters,iidmatrix);

%for k = 1:K
    %Uinit{k} = nvecs(Y,k,R(k));
%Xn = tenmat(X,k);
%Xn = matricize(double(Y),k)';
%%Xnn= Xn*Xn';
%[Uinit{k},~]=eigs(double(Xnn),3);
%end

if ~isempty(theta)
[ Sigma,S,invS] = createSigma2( n1,n2,theta,0,0);

else
    Sigma=iidmatrix;
end


U = Uinit;
Up = Uinit;

for k = 1:K
Up{k} =  (Sigma{k})\U{k};
end
nsample = size(X,1);

Up{K+1} = sqrtm(X/(X'*X)*X'+1e-8*eye(nsample));

fit = 1e3;

Ub = Up;
Ub{K+1} = X/(X'*X);

X1 = (1:n1)/n1;
X2 = (1:n2)/n2;

Y = tensor(Y);
normX = norm((Y));

for iter = 1:maxiters
    
    fitold = fit;
    
    % Iterate over all N modes of the tensor
    for n = 1:K
        Utilde = ttm((Y), Up, -n, 't');
        
        % Maximize norm(Utilde x_n W') wrt W and
        % keeping orthonormality of W
        
        Xn = real(double(matricize(Utilde,n))');
        WW = Xn*Xn';

        [Un,~] = eigs(invS{n}'*WW*invS{n}, R(n));
        U{n} = S{n}*Un;
        Up{n} = invS{n}^2*U{n};
        Ub{n} = Up{n};
    end
    
    % Assemble the current approximation
    core = ttm(Y, Ub, 't');


    
    % Compute fit`
    normresidual = sqrt( normX^2 - norm(core)^2 );
    fit = 1 - (normresidual / normX); %fraction explained by model
    fitchange = abs(fitold - fit);
%     
%             fprintf(' Iter %2d: fit = %e fitdelta = %7.1e\n', iter, fit, fitchange);

    % Check for convergence
    if (iter > 1) && (fitchange < 1e-9)
        break;
    end
    A = real(double(ttm(core,U,[1 2])));
    yhat = ttm(tensor(A),X,3);
    E = double(Y-yhat);
    if mod(iter,4) == 1 && isoptimize
        if isoptimizesigma
            mylikelihood  = @(x) -tensormle(E,X1,X2,x(1),x(1),x(2));
            para = fmincon(mylikelihood,[10,0]);
            theta = para(1);
            sigma = para(2);
            
        else
            mylikelihood  = @(x) -tensormle(E,X1,X2,x,x,0);
            para = fmincon(mylikelihood,10);
            theta = para(1);
            sigma=0;
        end
        
        [ Sigma,S,invS] = createSigma2( n1,n2,theta,sigma,isiid);
    end
    
    if isoptimize==0
        para=0;
    end
    
    

    %%

end



A = real(double(ttm(core,U,[1 2])));


end

