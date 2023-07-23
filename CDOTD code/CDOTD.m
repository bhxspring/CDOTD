function [G,W,obj] = CDOTD(X,St,W1,StartInd,nClusters,lambda)
% solve [G,W] = arg max  trace(W'*(Sb-lambda*St)*W) 
% obj:       the objective function value 
% X:         the d*n centered data matrix
% St:        the d*d total scatter matrix
% W1:        the d*m initial projection matrix
% StartInd:  50 randomly initialized cluster vectors
% nClusters: the number of clusters in the data set
% lambda:    the balance parameter
% The code is written by Quan Wang 

epsilon = 1e-8;
maxIter = 100;
m = size(W1,2);

W = W1;
G = full(ind2vec(StartInd',nClusters))';

it = 0;
obj_old = 0;
obj_new = 0.1;
while abs((obj_old-obj_new)/obj_new)>epsilon&&it<maxIter
    it = it+1;
    obj_old = obj_new;    
    
    U = W'*X;
    [G,~] = CDO(U,nClusters,G);  
    
    Sb = X*G*(G'*G)^-1*G'*X';
    M = Sb-lambda*St;
    
    [W,~,~] = eig1(M,m);
    obj_new = trace(W'*M*W);
    obj(it) = obj_new;
end
if it == maxIter
    disp(['Warnning: the CDOTD does not converge within ',num2str(maxIter),' iterations!']);
end
dbstop if error

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% subfunction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [G,ob] = CDO(U,nClusters,G)
% solve G = arg max trace( (G'*G)^(-1)*(G'*U'*U*G) ) 
% ob:        the objective function value
% U:         the m*n matrix
% nClusters: the number of clusters in the data set
% G:         the n*c cluster indicator matrix
% The code is written by Quan Wang

epsilon = 1e-8;
N = zeros(nClusters,1);
D = zeros(nClusters,1);
N0 = zeros(nClusters,1);
D0 = zeros(nClusters,1);
Nk = zeros(nClusters,1);
Dk = zeros(nClusters,1);
T = U*G;
for i=1:nClusters
    N(i) = T(:,i)'*T(:,i);
    D(i) = G(:,i)'*G(:,i);
end
iter = 2;
obj(1) = 1;
obj(2) = sum(N./D);
numPts = size(U,2);
Uii = zeros(numPts,1);
for i=1:numPts
    Uii(i) = U(:,i)'*U(:,i);
end
while abs((obj(iter-1)-obj(iter))/obj(iter))>epsilon
    iter = iter+1;    
    for i=1:numPts
        p = find(G(i,:)==1);        
        if D(p)>1
            G(i,p) = 0;
        else
            continue;
        end        
        for k=1:nClusters
            if k==p
                Nk(k) = N(k); 
                Dk(k) = D(k);
                N0(k) = N(k)-2*U(:,i)'*T(:,k)+Uii(i); 
                D0(k) = D(k)-1;
            else
                N0(k) = N(k); 
                D0(k) = D(k);
                Nk(k) = N(k)+2*U(:,i)'*T(:,k)+Uii(i); 
                Dk(k) = D(k)+1;
            end
        end
        delta = Nk./Dk-N0./D0;
        [~,q] = max(delta);
        G(i,q) = 1;
        if q~=p
            T(:,p) = T(:,p)-U(:,i);
            T(:,q) = T(:,q)+U(:,i);
            N(p) = N0(p); 
            D(p) = D0(p);
            N(q) = Nk(q); 
            D(q) = Dk(q);
        end
    end
    obj(iter) = sum(N./D);
end
iter = iter-2;
obj(1) = [];
ob = obj(end);
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
