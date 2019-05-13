function [ data ] = matricize( X,n )

cdims = n;
N = length(size(X));
if n<=N
    rdims = setdiff(1:N, cdims);
    tsize = size(X);
    data = reshape(permute(X,[rdims cdims]), [prod(tsize(rdims)), prod(tsize(cdims))]);
else
    data = X(:);
end

