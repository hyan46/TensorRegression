function [ likelihood ] = tensormle( E,X1,X2,theta1,theta2,sigma )
[n1,n2,K]= size(E);

Sigma1 = (exp(-theta1*squareform(pdist(X1')))+sigma^2*eye(n1));
Sigma2 =  (exp(-theta2*squareform(pdist(X2')))+sigma^2*eye(n2));

E1 = double(ttm(tensor(E),{inv(Sigma1),inv(Sigma2)},[1 2]));

likelihood = -1/2*n1*K*log(det(Sigma1))-1/2*n2*K*log(det(Sigma2)) -1/2*sum(sum(sum(E1.*E)));

end

