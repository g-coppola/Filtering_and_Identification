function [x_est, P_est] = UKF_augmented(f,h,y,x_prev,P_prev,Q,V,alpha,beta,kappa)
n = length(x_prev);
nw = size(Q,1);
nv = size(V,1);

L = n+nw+nv;

x_a = [x_prev; zeros(nw,1); zeros(nv,1)];
P_a = blkdiag(P_prev,Q,V);

lambda = alpha^2*(L+kappa)-L;

W_m = [lambda/(L+lambda), 1/(2*(L+lambda))*ones(1,2*L)];
W_c = W_m;
W_c(1) = lambda/(L+lambda) + (1-alpha^2+beta);

% Sigma Points
S = sqrtm(P_a);   
X = zeros(L,2*L+1);
X(:,1) = x_a;

for k = 1:L
    X(:,k+1) = x_a + sqrt(L+lambda)*S(:,k);
    X(:,k+L+1) = x_a - sqrt(L+lambda)*S(:,k);
end

% -- State Prediction --
X_k = zeros(n,2*L+1);
    
for k = 1:2*L+1
    w_k = X(n+1:n+nw,k);
    X_k(:,k) = f(X(1:n,k))+w_k;
end
    
% Mean (State)
x_k = X_k*W_m';
    
% Covariance (State)
P_k = zeros(n);
    
for k = 1:2*L+1
    P_k = P_k + W_c(k)*(X_k(:,k)-x_k)*(X_k(:,k)-x_k)';
end

% -- Measurement Prediction --
Y_k = zeros(nv,2*L+1);
    
for k = 1:2*L+1
    v_k = X(n+nw+1:end,k); 
    Y_k(:,k) = h(X_k(:,k)) + v_k;
end

% Mean (Output)
y_k = Y_k*W_m';
    
% Covariance (Output)
P_yy = zeros(nv);
for k = 1:2*L+1
    P_yy = P_yy + W_c(k)*(Y_k(:,k)-y_k)*(Y_k(:,k)-y_k)';
end

% Cross Covariance
P_xy = zeros(n,nv);
for k = 1:2*L+1
    P_xy = P_xy + W_c(k)*(X_k(:,k)-x_k)*(Y_k(:,k)-y_k)';
end

K = P_xy / P_yy;

x_est = x_k + K*(y-y_k);
P_est = P_k - K*P_yy*K';
    
end

