close all
clear


%% Lorenz 63 model

T = 5e4; %20e4 in the original experiments
Tobs = 5;
h = 2e-4;
s2x = h/10;
s2y = 1;

A = [10, 28, 8/3];
ko = 5;
H = eye(3); %H = H([1 3],:);
dimobs = size(H,1);
x = zeros(3,T);
y = zeros(dimobs,T);

x(:,1) = [-6; -5.5; 24.5] + sqrt(s2x)*randn(3,1);
y(:,1) = H*(ko*x(:,1)) + sqrt(s2y)*randn(dimobs,1);

for t = 2:T
   x(1,t) = x(1,t-1) + h*( A(1)*(x(2,t-1)-x(1,t-1)) ) + sqrt(s2x)*randn;
   x(2,t) = x(2,t-1) + h*( x(1,t-1)*A(2) - x(1,t-1)*x(3,t-1) - x(2,t-1) ) + sqrt(s2x)*randn;
   x(3,t) = x(3,t-1) + h*( x(1,t-1)*x(2,t-1) - A(3)*x(3,t-1) ) + sqrt(s2x)*randn;
   
   y(:,t) = H*(ko*x(:,t)) + sqrt(s2y)*randn(dimobs,1);
end

%% Config 

% To estimate (dimension of theta)
dim_theta = 3;
total_simulations= 1; % total number of simulations e.g., 10

% Plots config
figures_on = 1;
print_every = Tobs*20;

%% UKF with state augmentation

for iter = 1:total_simulations
    
    [UKF_NMSEx,UKF_NMSEparam,x_est,param_est,ttotal] = UKF_stateaugmentation_L63(x,y,s2x,s2y,h,T,Tobs,A,ko,H,dim_theta,iter,figures_on,print_every);
    
    fprintf(1,'----------------------------------------------------\n');
    fprintf(1,'UKF with state augmentation: iteration number %d\n', iter);
    fprintf(1,'----------------------------------------------------\n');
    fprintf(1,'- Time to finish = %7.4f min\n',ttotal);
    fprintf(1,'- Averaged NMSEx = %7.7f\n', mean(UKF_NMSEx(1:Tobs:T)) );
    fprintf(1,'- Averaged NMSEtheta = %7.7f\n', mean(UKF_NMSEparam(1:Tobs:T)) );    
    fprintf(1,'----------------------------------------------------\n\n');
    
end

fprintf(1,'\n');

%% EnKF with state augmentation

for iter = 1:total_simulations
    
    [EnKF_NMSEx,EnKF_NMSEparam,x_est,param_est,ttotal] = EnKF_stateaugmentation_L63(x,y,s2x,s2y,h,T,Tobs,A,ko,H,dim_theta,iter,figures_on,print_every);
    
    fprintf(1,'----------------------------------------------------\n');
    fprintf(1,'EnKF with state augmentation: iteration number %d\n', iter);
    fprintf(1,'----------------------------------------------------\n');
    fprintf(1,'- Time to finish = %7.4f min\n',ttotal);
    fprintf(1,'- Averaged NMSEx = %7.7f\n', mean(EnKF_NMSEx(1:Tobs:T)) );
    fprintf(1,'- Averaged NMSEtheta = %7.7f\n', mean(EnKF_NMSEparam(1:Tobs:T)) );    
    fprintf(1,'----------------------------------------------------\n\n');
    
end

fprintf(1,'\n');

%% Nested hybrid filter (NHF): SMC + EKFs (sequential Monte Carlo for theta estimation + extended Kalman filters for x estimation)

for iter = 1:total_simulations
    
   [NHF_NMSEx,NHF_NMSEparam,x_est,param_est,ttotal] = NHF__SMC_EKF_L63(x,y,s2x,s2y,h,T,Tobs,A,ko,H,dim_theta,iter,figures_on,print_every);
    
    fprintf(1,'----------------------------------------------------\n');
    fprintf(1,'NHF (SMC with EKFs): iteration number %d\n', iter);
    fprintf(1,'----------------------------------------------------\n');
    fprintf(1,'- Time to finish = %7.4f min\n',ttotal);
    fprintf(1,'- Averaged NMSEx = %7.7f\n', mean(NHF_NMSEx(1:Tobs:T)) );
    fprintf(1,'- Averaged NMSEtheta = %7.7f\n', mean(NHF_NMSEparam(1:Tobs:T)) );    
    fprintf(1,'----------------------------------------------------\n\n');
    
end

fprintf(1,'\n');


%% Nested Gaussian filter (NGF): UKF + EKFs (unscented Kalman filter for theta estimation + extended Kalman filters for x estimation)


for iter = 1:total_simulations
    
    [NGF_NMSEx,NGF_NMSEparam,x_est,param_est,ttotal] = NGF__UKF_EKF_L63(x,y,s2x,s2y,h,T,Tobs,A,ko,H,dim_theta,iter,figures_on,print_every);
    
    fprintf(1,'----------------------------------------------------\n');
    fprintf(1,'NGF (UKF with EKFs): iteration number %d\n', iter);
    fprintf(1,'----------------------------------------------------\n');
    fprintf(1,'- Time to finish = %7.4f min\n',ttotal);
    fprintf(1,'- Averaged NMSEx = %7.7f\n', mean(NGF_NMSEx(1:Tobs:T)) );
    fprintf(1,'- Averaged NMSEtheta = %7.7f\n', mean(NGF_NMSEparam(1:Tobs:T)) );    
    fprintf(1,'----------------------------------------------------\n\n');
    
end

fprintf(1,'\n');

