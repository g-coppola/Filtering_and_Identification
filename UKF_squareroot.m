function [x_est,S_est] = UKF_squareroot(f,h,y,x_prev,S_prev,Q,V,alpha,beta,kappa)
    n = length(x_prev);
    m = length(y);
    

    lambda = alpha^2*(n+kappa)-n;

    W_m = [lambda/(n+lambda), 1/(2*(n+lambda))*ones(1,2*n)];
    W_c = W_m;
    W_c(1) = lambda/(n+lambda) + (1-alpha^2+beta);
    
    % Sigma Points
    X = [x_prev, x_prev+sqrt(n+lambda)*S_prev, x_prev-sqrt(n+lambda)*S_prev];


    % -- State Prediction --
    X_k = zeros(n,2*n+1);
    
    for k = 1:2*n+1
        X_k(:,k) = f(X(:,k));
    end
    
    % Mean (State)
    x_k = X_k*W_m';
    
    % qr update for predicted covariance square root
    x_dev =  X_k - x_k;
    [~,S_k] = qr([sqrt(W_c(2:end)).*x_dev(:,2:end), sqrtm(Q)']',0);
    U = x_dev(:,1)*sqrt(W_c(1));

    if W_c(1) < 0
        S_pred = cholupdate(S_k,U,'-');
    else
        S_pred = cholupdate(S_k,U,'+');
    end


    % -- Measurement Prediction --
    Y_k = zeros(m,2*n+1);
    
    for k = 1:2*n+1
        Y_k(:,k) = h(X_k(:,k));
    end

    % Mean (Output)
    y_k = Y_k*W_m';
    
    % qr update for measurement covariance square root
    y_dev = Y_k - y_k;
    [~,S_y] = qr([sqrt(W_c(2:end)).*y_dev(:,2:end), sqrtm(V)']',0);
    U_y = y_dev(:,1)*sqrt(W_c(1));

    if W_c(1) < 0
        Sy_pred = cholupdate(S_y,U_y,'-');
    else
        Sy_pred = cholupdate(S_y,U_y,'+');
    end
    

    % Cross Covariance
    P_xy = zeros(n,m);
    for k = 1:2*n+1
        P_xy = P_xy + W_c(k)*(X_k(:,k)-x_k)*(Y_k(:,k)-y_k)';
    end

    K = (P_xy / Sy_pred') / Sy_pred;

    x_est = x_k + K*(y-y_k);
    U_k = K*Sy_pred;
    S_est = S_pred;

    for i = 1:size(U_k,2)
        S_est = cholupdate(S_est, U_k(:,i),'-');
    end
end

