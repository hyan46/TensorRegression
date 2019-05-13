 function [ alpha,Yhattr,Vhat,lambdanew,gcvscore,others] = tensorregmethod( X,Y,W,Xte,Vte,casemethod,options)
% casemethod: 1 tensor regression
% casemethod: 2 tensor decomposition regression
% casemethod: 3 PCA
% alpha: tensor regression coefficient
% Yhattr: Training prediciton
% Vhat:  Testing prediction
% lambanew: Selected lambda if any
% gcvscore: GCV criterion

n1 = size(Y,1);
n2 = size(Y,2);
ntr = size(Y,3);
others = [];
if isempty(options.periodic1)
    options.periodic1 = 0;
end
if isempty(options.periodic2)
    options.periodic2 = 0;
end
cperiodc = [options.periodic1,options.periodic2];

if isempty(options.smooth)
    issmooth  =0; 
else
    issmooth = options.smooth;
end


switch casemethod
    case 0 % Tensor regression with identity
        
    case 1 % Tensor regression
        k1 = options.k1;
        k2 = options.k2;
        lambda1 = options.lambda1;
       [ Sigma,S,invS] = createSigma2( n1,n2,options.theta1,0,0);

        
        B = cell(2,1);
        B{1} = bsplineBasis(n1,k1);
        B{2} = bsplineBasis(n2,k2);
        if options.periodic1 
            B{1} = periodBsplineBasis(n1,k1);
        end
        if options.periodic2 
            B{2} = periodBsplineBasis(n2,k2);
        end
        
        [ alpha,Yhattr,Vhat,lambdanew,gcvscore] = tensorreg(X,Y,W,Xte,B,lambda1,Sigma);
        
    case 2
        I1 = options.I1;
        I2 = options.I2;
        lambda2 = 0;
        
        if ~issmooth
            lambda2 = 0;
        end
            
        if ~isempty(lambda2) 
            lambdanew = lambda2;
            [gcvscore,T,Uinit] = myregtucker_als(tensor(Y),[I1,I2],50,lambda2,cperiodc);
            others.T = T;
            H = cell(3,1);
            for i = 1:2
                H{i} = (T.U{i}'*T.U{i})\T.U{i}';
            end
            if isempty(W)
                H{3} = (X'*X)\X';
            else
                H{3} = (X'*W*X)\(X'*W);
            end
            
            Str = ttm(tensor(Y),H,[1,2]);
            beta = double(ttm(Str,H{3},3));
            if ~isempty(Xte)
                Y1 = ttm(tensor(beta),Xte,3);
                Vhat = double(ttm(Y1,T.U,[1,2]));
            else
                Vhat  = [];
            end
            alpha = double(ttm(tensor(beta),T.U,[1,2]));
            Yhattr = double(ttm(tensor(alpha),X,3));
%            Indices = crossvalind('Kfold', ntr, 10);
%            gcvscore=gcv(lambda2);
        else
            [ T,alpha,Vhat,lambdanew ] = myregtucker_regcross(X,Y,I1,I2);
            gcvscore=0;
        end
        
    case 3 % PCA
        Y = double(Y);
        Ys = Y;
        
        
        if issmooth
            for i = 1:ntr
                Ys(:,:,i) = smoothn(Y(:,:,i));
            end
        end
        k3 = options.k3;
        V3 = matricize(Ys,3);
        [U3all,Eall]= svd(V3,'econ');
        U3 = U3all(:,1:k3);
        Xscore = U3*(U3'*V3);
        %         Xscore = V3;
        Xs = reshape(Xscore,n1,n2,ntr);
        H = ((X'*X)\X');
        alpha = double(ttm(tensor(Xs),H,3));
        Vhat = double(ttm(tensor(alpha),Xte,3));
        Yhattr = double(ttm(tensor(alpha),X,3));
        lambdanew=0;
        dEall = diag(Eall);
        gcvscore = sum(dEall(1:k3).^2)/trace(V3'*V3);
        
    case 4        
        Ys = Y;
        if issmooth
            for i = 1:ntr
                Ys(:,:,i) = smoothn(Y(:,:,i));
            end
        end
        if isempty(W)
            HX = ((X'*X)\X');
            H = (X/(X'*X)*X');
        else
            HX = (X'*W*X)\(X'*W);
            H = (X/(X'*W*X)*X'*W);
        end
        alpha = double(ttm(tensor(Ys),HX,3));
        if ~isempty(Xte)
            Vhat = double(ttm(tensor(alpha),Xte,3));
        else
            Vhat = [];
        end
        Yhattr = double(ttm(tensor(Ys),H,3));
        lambdanew=0;
        gcvscore = 0;
        
    case 5
        k3 = options.k3;
        V3 = matricize(Y,3);
        [U3all,Eall]= svd(V3,'econ');
        U3 = U3all(:,1:k3);
        Xscore = U3*(U3'*V3);
        %         Xscore = V3;
        Xs = reshape(Xscore,n1,n2,ntr);
        H = ((X'*X)\X');
        alpha = double(ttm(tensor(Xs),H,3));
        Vhat = double(ttm(tensor(alpha),Xte,3));
        lambdanew=0;
        dEall = diag(Eall);
        gcvscore = sum(dEall(1:k3).^2)/trace(V3'*V3);
        
    case 6
        Ys = Y;
        if isempty(W)
            HX = ((X'*X)\X');
            H = (X/(X'*X)*X');
        else
            HX = (X'*W*X)\(X'*W);
            H = (X/(X'*W*X)*X'*W);
        end

        alpha = double(ttm(tensor(Ys),HX,3));
        if ~isempty(Xte)
            Vhat = double(ttm(tensor(alpha),Xte,3));
        else
            Vhat = [];
        end
        Yhattr = double(ttm(tensor(Ys),H,3));
        lambdanew=0;
        gcvscore = 0;
    case 7
        % alpha: tensor regression coefficient
        % Yhattr: Training prediciton
        % Vhat:  Testing prediction
        % lambanew: Selected lambda if any
        % gcvscore: GCV criterion
        
        Ys = Y;
        R = options.R;
        if issmooth
            for i = 1:ntr
                Ys(:,:,i) = smoothn(Y(:,:,i));
            end
        end
        
        isoptimizesigma = options.isoptimizesigma;
        isoptimize = options.isoptimize;
        isiid = options.isiid;
        theta = options.theta7;
        [ alpha,U,para] = onesteptensoreg( Ys,X,R,isoptimize,isoptimizesigma,isiid,theta);
        Yhattr = double(ttm(tensor(alpha),X,3));
        others.U = U;
        if ~isempty(Xte)
            Vhat = double(ttm(tensor(alpha),Xte,3));
        else
            Vhat = [];
        end
        
        lambdanew = para;
        gcvscore = 0;
        
end


    function meanRSS = gcv(lambda2)
        RSS = zeros(10,1);
        for ii = 1:10
            Xtr = X(Indices~=ii,:);
            Ytr = Y(:,:,Indices~=ii);
            Xte = X(Indices==ii,:);
            Yte = Y(:,:,Indices==ii);
            [Yhat] = myregtucker_reg(I1,I2,lambda2,Xtr,Ytr,Xte);
            RSS(ii) = mse(Yhat(:)-Yte(:));
        end
        meanRSS = mean(RSS);
    end

end

