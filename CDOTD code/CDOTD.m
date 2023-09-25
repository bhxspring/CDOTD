function [G,W,obj] = CDOTD(X,St,W1,StartInd,nClusters,lambda)
% solve [G,W] = arg max  trace(W'*(Sb-lambda*St)*W) 
% obj:       the objective function value 
% X:         the d*n centered data matrix
% St:        the d*d total scatter matrix
% W1:        the d*m initial projection matrix
% StartInd:  one of the 50 randomly initialized cluster vectors
% nClusters: the number of clusters in the data set
% lambda:    the balance parameter
% The code is written by Quan Wang 

epsilon = 1e-8;
maxIter = 100;
m = size(W1,2);

W = W1;
G = full(ind2vec(StartInd',nClusters))';

obj = zeros(maxIter,1);
it = 0;
obj_old = 0;
obj_new = 0.1;
while abs((obj_old-obj_new)/obj_new)>epsilon&&it<maxIter
    it = it+1;
    obj_old = obj_new;    
    
    U = W'*X;
    [G,~] = CDO(U,G);  
    
    Sb = X*G*(G'*G)^-1*G'*X';
    M = Sb-lambda*St;
    
    [W,~,~] = eig1(M,m);
    obj_new = trace(W'*M*W);
    obj(it) = obj_new;
end
if it == maxIter
    disp('Warnning: the CDOTD does not converge within maxIter iterations!');
end
if it<maxIter
    obj(it+1:end) = [];
end
% dbstop if error

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% subfunction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [G,ob] = CDO(U,G)

T = U*G;
N = sum(T.*T,1)';
D = sum(G.*G,1)';
Uii = sum(U.*U,1)';

maxIter = 100;
epsilon = 1e-8;
obj = zeros(maxIter+2,1);
iter = 2;
obj(2) = sum(N./D);
while abs((obj(iter-1)-obj(iter))/obj(iter))>epsilon
    if iter>maxIter
        disp('Warnning: The maxIter iterations has reached in CDO!');
        break;
    end
    iter = iter+1;

    [G,N,D,T] = updateArrayG(G,N,D,U,T,Uii);

    obj(iter) = sum(N./D);
end
ob = obj(iter);
% iter = iter-2;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% subfunction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [eigvec, eigval, eigval_full] = eig1(A, c, isMax, isSym)

if nargin < 2
    c = size(A,1);
    isMax = 1;
    isSym = 1;
elseif c > size(A,1)
    c = size(A,1);
end;

if nargin < 3
    isMax = 1;
    isSym = 1;
end;

if nargin < 4
    isSym = 1;
end;

if isSym == 1
    A = max(A,A');
end;
[v d] = eig(A);
d = diag(d);
%d = real(d);
if isMax == 0
    [d1, idx] = sort(d);
else
    [d1, idx] = sort(d,'descend');
end;

idx1 = idx(1:c);
eigval = d(idx1);
eigvec = v(:,idx1);

eigval_full = d(idx);

end
