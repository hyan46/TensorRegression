function [ Sigma,S,invS] = createSigma( n1,n2,theta,sigma,isiid)
if isempty(theta)
    isiid = 1;
end
if ~isiid
    Xt = linspace(0,1,n1);
    Sigma{1} = (exp(-theta*squareform(pdist(Xt')))+sigma^2*eye(n1));
    Xt = linspace(0,1,n2);
    Sigma{2} = (exp(-theta*squareform(pdist(Xt')))+sigma^2*eye(n2));
else
    Sigma{1} = eye(n1);
    Sigma{2} = eye(n2);
end
S{1} = (sqrtm(Sigma{1}));
S{2} = (sqrtm(Sigma{2}));
invS{1} = inv(S{1});
invS{2} = inv(S{2});
end
