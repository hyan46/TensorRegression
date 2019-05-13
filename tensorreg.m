function [ alpha,Yhattr,Yhatte ,lambda,gcvscore] = tensorreg(X,Y,W,Xte,B,lambda,Sigma)

isAutoLambda = isempty(lambda);
ndim = 2;
D = cell(2,1);
H = cell(3,1);
p = size(X,2);
for i = 1:2
    ki = size(B{i},2);
    Di = diff(eye(ki),1);
    D{i} = Di'*Di;
end

if isempty(W)
    H{3} = (X/(X'*X)*X');
    if ~isempty(Xte)
        Hte3 = (Xte/(X'*X)*X');
    end
else
    H{3} = (X/(X'*W*X)*X'*W);
    if ~isempty(Xte)
        Hte3 = (Xte/(X'*W*X)*X'*W);
    end
end
    

df3 = p;
if ~isAutoLambda
    for i = 1:2
        H{i} = B{i}/(B{i}'/Sigma{i}*B{i}+lambda*D{i}'*D{i})*B{i}'/Sigma{i};
    end
else
    C = cell(2,1);
    U = cell(2,1);
    L = cell(2,1);
    Z = cell(2,1);
    for i = 1:2
        
        L{i} = sqrtm(B{i}'*B{i});
        L{i} = L{i} + 1e-8*eye(size(L{i}));
        [U{i},C{i}] =  svd((L{i}')\(D{i}'*D{i})/(L{i}));
        Z{i} = B{i}/(L{i}')*U{i};
    end
end
gcvscore=0;
if isAutoLambda
    [lambda,gcvscore]= fminbnd(@gcv,1e-2,1e3);
    for idim = 1:ndim
        H{idim} = Z{idim}*diag(1./(ones(size(C{idim},1),1) + lambda*diag(C{idim})))*Z{idim}';
    end
end

beta = double(ttm(tensor(Y),H,[1,2]));
alpha = double(ttm(tensor(beta),((X'*X)\X'),3));
Yhattr = double(ttm(tensor(beta),H,3));
if ~isempty(Xte)
    Yhatte = double(ttm(tensor(beta),Hte3,3));
else
    Yhatte = [];
end

    function GCVscore = gcv(lambda)
        % Search the smoothing parameter s that minimizes the GCV score
        %---
        dfi = zeros(ndim+1,1);
        for idim = 1:ndim
            %%
            H{idim} = Z{idim}*diag(1./(ones(size(C{idim},1),1) + lambda*diag(C{idim})))*Z{idim}';
            dfi(idim) = sum(1./(1+lambda*diag(C{idim})));
        end
        dfi(3) = df3;
        df = prod(dfi);
        Yhat = ttm(tensor(Y),H);

        errory = ((Y-Yhat).^2);
        RSS = sum(errory(:));
        n = numel(Y);
        GCVscore = RSS/n/(1-df/n)^2;
    end

end

