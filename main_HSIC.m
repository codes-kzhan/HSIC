clear all;  clear memory;
addpath('./Utility')

load ImageNet10Cls_Used

[dim, N] = size(X{1});
MaxIter = 9;
innerMax = 10;
r = 5;            % r is the power of alpha_i
L = 128;          % Hashing code length
beta = .01;       % Hyper-para beta
gamma = .001;     % Hyper-para gamma
lambda = 0.00001; % Hyper-para lambda
rate  = 0.2;

%------------Parameter Initialization--------------
rand('seed',250);
r_sample = X{1}(:,randsample(N, 500),:);
[pcaW, ~] = eigs(cov(r_sample'), L);
B = sign(pcaW'*X{1});% B = sign(randn(L,N));

n_cluster = numel(unique(gnd));
viewNum = size(X,2);
alpha = ones(viewNum,1) / viewNum;
W = cell(1,viewNum);

rand('seed',200);
C = B(:,randsample(N, n_cluster));
HamDist = 0.5*(L - B'*C);
[~,ind] = min(HamDist,[],2);
G = sparse(ind,1:N,1,n_cluster,N,N);
CG = C*G;
%------------End Initialization--------------

XXT = cell(1,viewNum);
for view = 1:viewNum
    XXT{view} = X{view}*X{view}';
end
D = sparse(diag(ones(L, 1))); % L2 norm version

for iter = 1:MaxIter
%     disp(sprintf('Iteration: %i', iter));
    %---------Seperate Bs and Bi--------------
    Bs = B(1:ceil(rate*L),:);
    Bi = B(ceil(rate*L)+1:end,:);
    
    %---------Update W--------------
    alpha_r = alpha.^r;
    WTX = zeros(L,N); 
    Wi = cell(1,viewNum);
    
    A = zeros(dim);
    T = zeros(dim,N);
    for v = 1:viewNum
        A = A + (1-gamma)*alpha_r(v)*XXT{v};
        T = T + alpha_r(v)*X{view};
    end
    
    Ws = (A+beta*eye(dim))\(T*Bs');
    for v = 1:viewNum
        Wi{v} = ((1-gamma)*XXT{v}+beta*eye(dim))\(X{v}*Bi');
        W{v} = [Wi{v} Ws];
        WTX  = WTX+alpha_r(v)*W{v}'*X{v};
    end

    %---------Update B--------------
    B = sign(WTX+lambda*CG);B(B==0) = -1;
    
    %---------Update C and G--------------
    for time = 1:innerMax
        DB = D*B; %C = ones(L,n_cluster); 
        % C(DB*G'<0) = -1;
        rho = 1e-3; mu = .2;
        for iter_num = 1:3
            grad = -DB*G'+ rho*repmat(sum(C),L,1);
            C = sign(C - 1/mu*grad);
        end

        HamDist = 0.5*(L - DB'*C);
        [~,indx] = min(HamDist,[],2);
        G = sparse(indx,1:N,1,n_cluster,N,N);
        
        CG = C*G;
        E = B - CG;
        Ei2 = sqrt(sum(E.*E, 2) + eps);
        D = sparse(diag(0.5./Ei2));
    end
    
    %---------Update alpha--------------
    h = zeros(viewNum,1);
    for view = 1:viewNum
        h(view) = norm(B-W{view}'*X{view},'fro')^2 + beta*norm(W{view},'fro')^2-gamma*norm(W{view}'*X{view},'fro');
    end
    H = bsxfun(@power,h, 1/(1-r));
    alpha = bsxfun(@rdivide,H,sum(H));
end

[~,label] = max(G,[],1);
res_cluster = ClusteringMeasure(gnd, label);
[fm, Precision, Recall] = compute_f(gnd, label'); 
fprintf('ACC = %.4f and NMI = %.4f, Purity = %.4f, F-Score = %.4f\n\n',res_cluster(1),res_cluster(2),res_cluster(3),fm);
