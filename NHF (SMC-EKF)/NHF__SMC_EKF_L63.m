function [NHF_NMSEx,NHF_NMSEtheta,x_est,theta_est,ttotal] = NHF__SMC_EKF_L63(x,y,s2x,s2y,h,T,Tobs,A,ko,H,dim_theta,iter,figures_on,print_every)
% NHF__SMC_EKF_L63 Nested hybrid filter, implementation with sequential 
% Monte Calor (SMC) filter in the parameter layer and a bank of extended 
% Kalman filters in the state layer. We estimate the evolution of a 
% stochastic Lorenz 63 model
%
%   Input variables:
%   x : state (ground truth)
%   y : synthetic observations
%   s2x : state variance 
%   s2y : observation variance
%   h : integration step
%   T : total time of simulation (number of discrete-time steps)
%   Tobs : time-steps between observations
%   A : theta (true parameters of the model)
%   ko : a parameter of the observation function
%   H : observation matrix 
%   dim_theta : number of parameters to estimate
%   iter : iteration number that runs now
%   figures_on : to decide whether to show figures or not
%   print_every : 


t0=clock;

% Dimension of x, theta and y
dim_x = size(x,1);
dim_y = size(H,1);

% Number of particles
M = 120;

% Particle filter parameters
theta_range = [5 25; 20 40; 0 10];
s2M = [2 8 1/5 1/2 0.04 0.04]./(M^1.5); % variance for the jittering (different per each parameter) [F C B H A1 A2]

% Estimates
theta_est = zeros(dim_theta,T);
x_est = zeros(dim_x,T);

% error
NHF_NMSEx = zeros(1,T);
NHF_NMSEtheta = zeros(1,T);

% Weights
W_mc = ones([M 1])/M;


% Initialization of theta and x
% theta
xinit = [-6; -5.5; 24.5] + randn(3,1);
Px_init = 2*eye(dim_x); 
Xm_mc = xinit*ones([1 M]);
for m = 1:M
    Px_mc{m} = Px_init;
end %m
theta_mc = zeros(dim_theta,M);
theta_mc(1,1:M) = theta_range(1,1) + ( theta_range(1,2) - theta_range(1,1) ) .* rand([1 M]);
theta_mc(2,1:M) = theta_range(2,1) + ( theta_range(2,2) - theta_range(2,1) ) .* rand([1 M]);
theta_mc(3,1:M) = theta_range(3,1) + ( theta_range(3,2) - theta_range(3,1) ) .* rand([1 M]);


% Sequential filtering
for t = 2:T
    % Every time there is a new observation (every Tobs steps)
    if mod(t,Tobs)==1
        
        %Jittering
        theta_mc(1,1:M) = rnd_tgaussian(theta_mc(1,1:M),s2M(1)*ones([1 M]),theta_range(1,1),theta_range(1,2));
        theta_mc(2,1:M) = rnd_tgaussian(theta_mc(2,1:M),s2M(3)*ones([1 M]),theta_range(2,1),theta_range(2,2));
        theta_mc(3,1:M) = rnd_tgaussian(theta_mc(3,1:M),s2M(3)*ones([1 M]),theta_range(3,1),theta_range(3,2));
        
        % Weights (second layer)
        loglik = 100*ones(M,1);
        MSE_EKFs = zeros(M,1);
        % for each point in the UKF
        for m = 1:M             
            [x_aux,MSE_aux,loglik_aux,Px_est] = EKFt0_tL63(x,y,Xm_mc(:,m),Px_mc{m},t-Tobs,t,Tobs,s2x,s2y,H,ko,h,theta_mc(:,m));
            Xm_mc(:,m) = x_aux(:,end);
            MSE_EKFs(m) = mean(MSE_aux);
            loglik(m) = -loglik_aux(end);
            Px_mc{m} = Px_est;
        end
        
        % Weight normalisation
        Wu_mc = exp( real(loglik) - max(real(loglik)) );
        W_mc = Wu_mc ./ sum(Wu_mc);        

        % State estimation para los dos primeros osciladores
        x_est(:,t) = Xm_mc(:,1:M)*W_mc;
        theta_est(:,t) = theta_mc*W_mc;
   
        % Error
        NHF_NMSEx(t) = sum((x(:,t)-x_est(:,t)).^2)./sum((x(:,t)).^2);   
        NHF_NMSEtheta(t)=sum((A'-theta_est(:,t)).^2)./sum((A').^2);
             
        % Resampling
        idx_mc = randsample(1:M,M,true,W_mc);
        theta_mc = theta_mc(:,idx_mc);
        Xm_mc = Xm_mc(:,idx_mc);
        Px_mc_old = Px_mc;
        for m = 1:M
            Px_mc{m} = Px_mc_old{idx_mc(m)};
        end %m
 
        % Figures
        if figures_on == 1
            if mod(t,print_every)==1

                figure(3)
                for j = 1:3
                    % Slow variables
                    subplot(2,3,j),
                    plot(1+Tobs:Tobs:t, x_est(j,1+Tobs:Tobs:t),'b'), hold on
                    plot(1+Tobs:Tobs:t, x(j,1+Tobs:Tobs:t),'k')
                    hold off

                end

                % Parameters
                subplot(2,3,4), 
                plot(1+Tobs:Tobs:t, theta_est(1,1+Tobs:Tobs:t),'b'),
                hold on,         
                plot(1+Tobs:Tobs:t, A(1)*ones(size(1+Tobs:Tobs:t)),'b--'), 
                plot(1+Tobs:Tobs:t, theta_est(2,1+Tobs:Tobs:t),'r'),
                plot(1+Tobs:Tobs:t, A(2)*ones(size(1+Tobs:Tobs:t)),'r--'), 
                plot(1+Tobs:Tobs:t, theta_est(3,1+Tobs:Tobs:t),'g'),
                plot(1+Tobs:Tobs:t, A(3)*ones(size(1+Tobs:Tobs:t)),'g--'), 
                hold off

                % NMSE
                subplot(2,3,5)
                semilogy(NHF_NMSEtheta(1:Tobs:t))
                ylabel('NMSE_\theta')
                subplot(2,3,6)
                semilogy(NHF_NMSEx(1:Tobs:t))
                ylabel('NMSE_x')
                
                
                sgtitle('NHF (SMC-EKF) - Lorenz 63 model')
                pause(0.1)

            end %mod(t,print_every)
               
        end % end figures   
        
        
    end % mod(t,Tobs)

end


% total time of simulation 
ttotal = etime(clock,t0)/60;

% Save data
clearvars -except NHF_NMSEtheta NHF_NMSEx x_est theta_est dim_y Tobs iter A x y H ttotal M
etiq_save = sprintf('data/NHF_SMCEKF_dimobs%d_iter%d.mat',dim_y, iter);
save(etiq_save);