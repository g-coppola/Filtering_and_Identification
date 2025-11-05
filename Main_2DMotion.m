%% 2D MOTION
clear
close all
clc

rng(1)

% Step
N = 150;

% Parameters
n = 4;
m = 2;
dt = 1;

% Accurate Sensors
%Q = diag([0.01, 0.01, 0.005, 0.005]); % More noise on position and velocity
%V = diag([0.05, 0.05]);  

% Not Accurate Sensors
%Q = diag([0.01, 0.01, 0.005, 0.005]);
%V = diag([0.5, 0.5]);  

% Stress Test
Q = diag([0.2, 0.2, 0.1, 0.1]);
V = diag([1, 1]);

alpha = 1e-3;
beta = 2;
kappa = 0;

% Dynamical Model

f = @(x) [x(1) + x(3)*dt + 0.1*sin(x(2));
         x(2) + x(4)*dt + 0.1*cos(x(1));
         x(3);
         x(4)];

h = @(x) [x(1); 
          x(2)];

%% MAIN LOOP
% Initial state
x_true = [0; 0; 1; 0.5];
x_est = [0; 0; 0.8; 0.3];
x_est_aug = x_est;
x_est_sr = x_est;

% Storage
X_true = zeros(n,N);
X_true(:,1) = x_true;
X_est = zeros(n,N);
X_est(:,1) = x_est;
X_est_aug = zeros(n,N);
X_est_aug(:,1) = x_est_aug;
X_est_sr = zeros(n,N);
X_est_sr(:,1) = x_est_sr;
Y = zeros(m,N);

P = diag([1 1 0.5 0.5]);
P_aug = P;
S = chol(P,'lower');

% Simulation
for k = 2:N
    % True State
    x_true = f(x_true) + mvnrnd(zeros(n,1),Q)';

    % Measurement
    y = h(x_true) + mvnrnd(zeros(m,1),V)';
    y_a = h(x_true);

    % UKF Non-Augmented
    [x_est, P] = UKF(f,h,y,x_est,P,Q,V,alpha,beta,kappa);

    % UKF Augmented
    [x_est_aug, P_aug] = UKF_augmented(f,h,y_a,x_est_aug,P_aug,Q,V,alpha,beta,kappa);

    % Square Root UKF
    [x_est_sr,S] = UKF_squareroot(f,h,y,x_est_sr,S,Q,V,alpha,beta,kappa);

    X_true(:,k) = x_true;
    X_est(:,k) = x_est;
    X_est_aug(:,k) = x_est_aug;
    X_est_sr(:,k) = x_est_sr;
    Y(:,k) = y;
end

figure(1)
subplot(2,2,[1 2])
plot(X_true(1,:),X_true(2,:),'-b','LineWidth',1.5)
hold on
plot(X_est(1,:),X_est(2,:),'--r','LineWidth',1.5);
plot(Y(1,:),Y(2,:),'.g','LineWidth',4);
legend('True Position','Estimated Position','Measurements')
xlabel('x'); ylabel('y')
grid

subplot(2,2,3)
plot(1:N,X_true(3,:),'-b','LineWidth',1.5)
hold on
plot(1:N,X_est(3,:),'--r','LineWidth',1.5);
legend('True Velocity [v_x]','Estimated v_x')
xlabel('time [s]'); ylabel('v_x')
grid

subplot(2,2,4)
plot(1:N,X_true(4,:),'-b','LineWidth',1.5)
hold on
plot(1:N,X_est(4,:),'--r','LineWidth',1.5);
legend('True Velocity [v_y]','Estimated v_y')
xlabel('time [s]'); ylabel('v_y')
grid

sgtitle('Estimation Problem with UKF Non-Augmented')

input("PRESS!")

%% UKF AUGMENTED

figure(2)
subplot(2,2,[1 2])
plot(X_true(1,:),X_true(2,:),'-b','LineWidth',1.5)
hold on
plot(X_est_aug(1,:),X_est_aug(2,:),'--k','LineWidth',1.5);
plot(Y(1,:),Y(2,:),'.g','LineWidth',4);
legend('True Position','Estimated Position','Measurements')
xlabel('x'); ylabel('y')
grid
hold off

subplot(2,2,3)
plot(1:N,X_true(3,:),'-b','LineWidth',1.5)
hold on
plot(1:N,X_est_aug(3,:),'--k','LineWidth',1.5);
legend('True Velocity [v_x]','Estimated v_x')
xlabel('time [s]'); ylabel('v_x')
grid
hold off

subplot(2,2,4)
plot(1:N,X_true(4,:),'-b','LineWidth',1.5)
hold on
plot(1:N,X_est_aug(4,:),'--k','LineWidth',1.5);
legend('True Velocity [v_y]','Estimated v_y')
xlabel('time [s]'); ylabel('v_y')
grid
hold off

sgtitle('Estimation Problem with UKF Augmented')

input('PRESS!')

%% SQUARE ROOT UKF

figure(3)
subplot(2,2,[1 2])
plot(X_true(1,:),X_true(2,:),'-b','LineWidth',1.5)
hold on
plot(X_est_sr(1,:),X_est_sr(2,:),'--','Color',[0.7 0.3 0.1],'LineWidth',1.5);
plot(Y(1,:),Y(2,:),'.g','LineWidth',4);
legend('True Position','Estimated Position','Measurements')
xlabel('x'); ylabel('y')
grid
hold off

subplot(2,2,3)
plot(1:N,X_true(3,:),'-b','LineWidth',1.5)
hold on
plot(1:N,X_est_sr(3,:),'--','Color',[0.7 0.3 0.1],'LineWidth',1.5);
legend('True Velocity [v_x]','Estimated v_x')
xlabel('time [s]'); ylabel('v_x')
grid
hold off

subplot(2,2,4)
plot(1:N,X_true(4,:),'-b','LineWidth',1.5)
hold on
plot(1:N,X_est_sr(4,:),'--','Color',[0.7 0.3 0.1],'LineWidth',1.5);
legend('True Velocity [v_y]','Estimated v_y')
xlabel('time [s]'); ylabel('v_y')
grid
hold off

sgtitle('Estimation Problem with Square-Root UKF')

input('PRESS!')
close all
%% COMPARISON PLOTS
figure(1)
subplot(2,2,[1 2])
plot(X_true(1,:),X_true(2,:),'--b','LineWidth',1)
hold on
plot(X_est(1,:),X_est(2,:),'-r','LineWidth',2)
plot(X_est_aug(1,:),X_est_aug(2,:),'-k','LineWidth',2);
plot(Y(1,:),Y(2,:),'.g','LineWidth',4);
legend('True Position','Estimated Position (UKF Non-Augmented)','Estimated Position (UKF Augmented)','Measurements')
xlabel('x'); ylabel('y')
grid
hold off

subplot(2,2,3)
plot(1:N,X_true(3,:),'--b','LineWidth',1)
hold on
plot(1:N,X_est(3,:),'-r','LineWidth',2);
plot(1:N,X_est_aug(3,:),'-k','LineWidth',2);
legend('True Velocity [v_x]','Estimated v_x (UKF Non-Augmented)','Estimated v_x (UKF Augmented)')
xlabel('time [s]'); ylabel('v_x')
grid
hold off

subplot(2,2,4)
plot(1:N,X_true(4,:),'--b','LineWidth',1)
hold on
plot(1:N,X_est(4,:),'-r','LineWidth',2);
plot(1:N,X_est_aug(4,:),'-k','LineWidth',2);
legend('True Velocity [v_y]','Estimated v_y (UKF Non-Augmented)','Estimated v_y (UKF Augmented)')
xlabel('time [s]'); ylabel('v_y')
grid
hold off

sgtitle('Comparison between UKF Non-Augmented and Augmented')

input("PRESS")

figure(2)
subplot(2,2,[1 2])
plot(X_true(1,:),X_true(2,:),'--b','LineWidth',1)
hold on
plot(X_est(1,:),X_est(2,:),'-r','LineWidth',2)
plot(X_est_sr(1,:),X_est_sr(2,:),'-','Color',[0.7 0.3 0.1],'LineWidth',2);
plot(Y(1,:),Y(2,:),'.g','LineWidth',4);
legend('True Position','Estimated Position (Standard UKF)','Estimated Position (Square-Root UKF)','Measurements')
xlabel('x'); ylabel('y')
grid
hold off

subplot(2,2,3)
plot(1:N,X_true(3,:),'--b','LineWidth',1)
hold on
plot(1:N,X_est(3,:),'-r','LineWidth',2);
plot(1:N,X_est_sr(3,:),'-','Color',[0.7 0.3 0.1],'LineWidth',2);
legend('True Velocity [v_x]','Estimated v_x (Standard UKF)','Estimated v_x (Square-Root UKF)')
xlabel('time [s]'); ylabel('v_x')
grid
hold off

subplot(2,2,4)
plot(1:N,X_true(4,:),'--b','LineWidth',1)
hold on
plot(1:N,X_est(4,:),'-r','LineWidth',2);
plot(1:N,X_est_sr(4,:),'-','Color',[0.7 0.3 0.1],'LineWidth',2);
legend('True Velocity [v_y]','Estimated v_y (Standard UKF)','Estimated v_y (Square-Root UKF)')
xlabel('time [s]'); ylabel('v_y')
grid
hold off

sgtitle('Comparison between Standard UKF and Square-Root UKF')
