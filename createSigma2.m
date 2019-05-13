function [ Sigma,S,invS] = createSigma2( n1,n2,theta,sigma,isiid)
if isempty(theta)
    isiid = 1;
end
if ~isiid
    Xt = linspace(0,1,n1);
    S{1} = (exp(-theta*squareform(pdist(Xt')))+sigma^2*eye(n1));
    Xt = linspace(0,1,n2);
    S{2} = (exp(-theta*squareform(pdist(Xt')))+sigma^2*eye(n2));
    
    
else
    S{1} = eye(n1);
    S{2} = eye(n2);
end
Sigma{1} = S{1}^2;
Sigma{2} = S{2}^2;
   
invS{1} = inv(S{1});
invS{2} = inv(S{2});
end
