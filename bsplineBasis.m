function [ B ] = bsplineBasis(n,k,sd,bd)
% bsplineBasis: Construct k Bspline Basis with n gridded with spline degree sd
% Input Variable: 
% n: length of signal or number of pixels 
% k: number of knots, k = n, sd must be zero, then I matrix
% sd: spline degree, sd = 0, then constant function
% bd: how many basis in boundary. 
% Coding by Hao Yan

if nargin < 4
    if nargin < 3
        sd = 3;
    end
    bd = sd - 1;
    if sd == 0
        bd = 0;
    end
end

if n == k 
    B = eye(n);
elseif sd == 0 && n~=k
    nk = floor(n/k);
    B = kron(eye(nk),ones(k,1))/sqrt(k);
    leftnk = mod(n,k);
    if leftnk
        B = blkdiag(B,ones(leftnk,1));
    end
else
    knots = [zeros(1,bd) linspace(0,n,k) n * ones(1,bd)];
    nKnots = length(knots) - sd;
    kspline = spmak(knots,eye(nKnots));
    B=spval(kspline,1:n)';
end


end

