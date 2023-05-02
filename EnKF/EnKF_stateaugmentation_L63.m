function [EnKF_NMSEx,EnKF_NMSEparam,x_est,theta_est,ttotal] = EnKF_stateaugmentation_L63(x,y,s2x,s2y,h,T,Tobs,A,ko,H,dim_theta,iter,figures_on,print_every)
% UKF_stateaugmentation_L63 Unscented Kalman filter with state augmentation
% (including x and theta in the state vector) in order to estimate the
% evolution of a stochastic Lorenz 63 model
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
totaldim = dim_theta+dim_x;
dim_y = size(H,1);


% Number of samples
N = totaldim;
% Parameter range of values
Param_range = [7 13; 25 30; 1 4]; 


% Init. variables
statetheta_est = zeros(totaldim,T);      % estimations of x and theta
EnKF_NMSEx = zeros(1,T);
EnKF_NMSEparam = zeros(1,T);



% Initialization of theta and x
% theta
Param_upd = Param_range(:,1)*ones(1,N) + (Param_range(:,2)-Param_range(:,1))*ones(1,N).*rand(dim_theta,N);
% x
xinit = [-6; -5.5; 24.5] + randn(3,1);
X_upd = xinit*ones(1,N) + 0.1*randn(dim_x,N);
% theta and x
statetheta_est(dim_theta+1:end,1) = xinit;

% Sequential filtering
for t = 2:T
   % Every time there is a new observation (every Tobs steps)
    if mod(t,Tobs)==1
        
        %PREDICTION
        % Artificial dynamics in theta
        Param_pred = Param_upd + sqrt(0.01)*randn(dim_theta,N);
        
        % Propagate the state
        X_pred = zeros(dim_x,N);
        for k = 1:Tobs
 
            X_pred(1,:) = X_upd(1,:) + h.*( Param_pred(1,:).*(X_upd(2,:)-X_upd(1,:)) ) + sqrt(s2x).*randn(1,N);
            X_pred(2,:) = X_upd(2,:)  + h.*( X_upd(1,:).*Param_pred(2,:) - X_upd(1,:).*X_upd(3,:) - X_upd(2,:) ) + sqrt(s2x).*randn(1,N); 
            X_pred(3,:) = X_upd(3,:) + h.*( X_upd(1,:).*X_upd(2,:) - Param_pred(3,:).*X_upd(3,:) ) + sqrt(s2x).*randn(1,N);    
            
            X_upd = X_pred;
                        
        end
        
        stateparam_pred = [Param_pred; X_pred];
         
        % UPDATE       
        Y = ko.*X_pred(diag(H)>0,:);
        
        % cov and cross_cov
        obs_mean = mean(Y,2);
        stateparam_mean = mean(stateparam_pred,2);
        aux_obs_cov = Y - obs_mean*ones(1,N);
        aux_thetastate_cov = stateparam_pred - stateparam_mean*ones(1,N);
        
        cross_cov = aux_thetastate_cov*aux_obs_cov'./(N-1);
        R_cov = s2y*eye(dim_y);
        obs_cov = aux_obs_cov*aux_obs_cov'./(N-1) + R_cov;
        
        % Kalman gain and update
        Kalman_gain = cross_cov/obs_cov;
        statetheta_upd = stateparam_pred +  Kalman_gain*(y(:,t)*ones(1,N) + sqrt(s2y)*randn(dim_y,N) - Y);
        Param_upd = statetheta_upd(1:dim_theta,:);
        X_upd = statetheta_upd(1+dim_theta:end,:);
             
        % Estimates
        statetheta_est(:,t) = mean(statetheta_upd,2);
   
        %NMSE
        EnKF_NMSEparam(t) = sum((A'-statetheta_est(1:dim_theta,t)).^2)./sum((A').^2);
        EnKF_NMSEx(:,t) = sum((x(:,t)-statetheta_est(dim_theta+1:end,t)).^2)./sum((x(:,t)).^2);
   
 
        % Figures
        if figures_on == 1
            if mod(t,print_every)==1
            
                figure(2)
                for j = 1:3
                    % Slow variables
                    subplot(2,3,j),
                    plot(1+Tobs:Tobs:t, statetheta_est(3+j,1+Tobs:Tobs:t),'b'), hold on
                    plot(1+Tobs:Tobs:t, x(j,1+Tobs:Tobs:t),'k')
                    hold off

                end

                % Parameters
                subplot(2,3,4), 
                plot(1+Tobs:Tobs:t, statetheta_est(1,1+Tobs:Tobs:t),'b'),
                hold on,         
                plot(1+Tobs:Tobs:t, A(1)*ones(size(1+Tobs:Tobs:t)),'b--'), 
                plot(1+Tobs:Tobs:t, statetheta_est(2,1+Tobs:Tobs:t),'r'),
                plot(1+Tobs:Tobs:t, A(2)*ones(size(1+Tobs:Tobs:t)),'r--'), 
                plot(1+Tobs:Tobs:t, statetheta_est(3,1+Tobs:Tobs:t),'g'),
                plot(1+Tobs:Tobs:t, A(3)*ones(size(1+Tobs:Tobs:t)),'g--'), 
                hold off

                % NMSE
                subplot(2,3,5)
                semilogy(EnKF_NMSEparam(1:Tobs:t))
                ylabel('NMSE_\theta')
                subplot(2,3,6)
                semilogy(EnKF_NMSEx(1:Tobs:t))
                ylabel('NMSE_x')
                
               
                sgtitle('EnKF (state augm.) - Lorenz 63 model')
                pause(0.1)
                
            end % mod(t,Tobs)==1
               
        end % end figures   
        
        
    end % mod(t,Tobs)

end%t

% Separate x and theta estiamtes in two variables
x_est = statetheta_est(1+dim_theta:end,:);
theta_est = statetheta_est(1:dim_theta,:);

% total time of simulation 
ttotal = etime(clock,t0)/60;

% Save data
clearvars -except EnKF_NMSEparam EnKF_NMSEx AXest dim_y Tobs x_est theta_est iter A x y H ttotal
etiq_save = sprintf('data/EnKF_dimobs%d_iter%d.mat',dim_y, iter);
save(etiq_save);

end
    


