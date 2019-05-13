 function [core,A,U] = tucker_onesetp(X,Y,R,maxiters,Sigma)

% Extract number of dimensions and norm of X.
N = ndims(Y)-1;

S{1} = (sqrtm(Sigma{1}));
S{2} = (sqrtm(Sigma{2}));
invS{1} = inv(S{1});
invS{2} = inv(S{2});

if numel(R) == 1
    R = R * ones(N,1);
end

Uinit = cell(N+1,1);

for n = 1:N
%     Uinit{n} = nvecs(X,n,R(n));
%Xn = tenmat(X,n);
Xn = matricize(double(Y),n)';
Xnn= Xn*Xn';
[Uinit{n},~]=eigs(double(Xnn),3);
end


U = Uinit;
Up = Uinit;

for n = 1:N
Up{n} =  (Sigma{n})\U{n};
end

Up{N+1} = sqrtm(X/(X'*X)*X');
fit = 1e3;

Ub = Up;
Ub{N+1} = X/(X'*X);

%%
normX = norm(Y);
for iter = 1:maxiters
    
    fitold = fit;
    
    % Iterate over all N modes of the tensor
    for n = 1:N
        Utilde = ttm(Y, Up, -n, 't');
        
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
    
end

A = real(double(ttm(core,U,[1 2])));




end


