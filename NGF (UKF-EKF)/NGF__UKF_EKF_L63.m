function [NGF_NMSEx,NGF_NMSEtheta,x_est,theta_est,ttotal] = NGF__UKF_EKF_L63(x,y,s2x,s2y,h,T,Tobs,A,ko,H,dim_theta,iter,figures_on,print_every)
% NGF__UKF_EKF_L63 Nested Gaussian filter, implementation with Unscented 
% Kalman filter in the parameter layer and a bank of extended Kalman
% filters in the state layer. We estimate the evolution of a 
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


% Important parameters of the algorithm
lambda = 1e-3;


t0=clock;

% Dimension of x, theta and y
dim_x = size(x,1);
dim_y = size(H,1);


% Number of sigma points
M = 2*dim_theta+1;
% Weights
ki = 1;
W = zeros(1,M);
W(1) =  ki/(dim_theta+ki);
W(2:end) = 1/(2*(dim_theta+ki));



% Init. variables
theta_est = zeros(dim_theta,T);
x_est = zeros(dim_x,T);
NGF_NMSEx = zeros(1,T);
NGF_NMSEtheta = zeros(1,T);


% Initialization of theta and x
% theta
tht_points = zeros(dim_theta,M);
tht_points_prev = tht_points;
Ptht_est = zeros(dim_theta,dim_theta,T);
epsilon = [3 1 0.5]; 
theta_mu_init = A - epsilon + 2*epsilon.*rand(1,3);
theta_est(:,1) = theta_mu_init' + randn(dim_theta,1);
Ptht = 1*eye(dim_theta); % cov. matrix of theta

% x
Px_init = zeros(dim_x,dim_x,M);
for m = 1:M
    Px_init(:,:,m) = 2*eye(dim_x);
end
xinit = [-6; -5.5; 24.5] + randn(3,1);
x_est(:,1) = xinit;
Xpoints = zeros(dim_x,M);         


% Sequential filtering
for t = 2:T
    
    % Every time there is a new observation (every Tobs steps)
    if mod(t,Tobs)==1
        
        %PREDICTION
        % Generate sigma-points
        tht_points_prev = tht_points;
        S = chol((dim_theta+ki)*Ptht);
        tht_points = theta_est(:,t-Tobs)*ones(1,M);
        tht_points(:,2:(dim_theta+1)) = tht_points(:,2:(dim_theta+1)) + S';
        tht_points(:,(dim_theta+2):end) = tht_points(:,(dim_theta+2):end) - S';
        
        % Weights (second layer)
        loglik = 100*ones(M,1);
        MSE_EKFs = zeros(M,1);
        euc_dist_tht_points = sqrt(sum((tht_points - tht_points_prev).^2));
        % for each point in the UKF
        for m = 1:M   
            if euc_dist_tht_points < lambda*sqrt(sum((tht_points_prev).^2))
                [x_aux,MSE_aux,loglik_aux,Px_est] = EKFt0_tL63(x,y,Xpoints(:,m),Px_init(:,:,m),t-Tobs,t,Tobs,s2x,s2y,H,ko,h,tht_points(:,m));
                Xpoints(:,m) = x_aux(:,end);
                MSE_EKFs(m) = mean(MSE_aux);
                loglik(m) = loglik_aux(end);
                Px_init(:,:,m) = Px_est;

            else
                [x_aux,MSE_aux,loglik_aux,Px_est] = EKFt0_tL63(x,y,xinit,2*eye(dim_x),1,t,Tobs,s2x,s2y,H,ko,h,tht_points(:,m));
                Xpoints(:,m) = x_aux(:,end);
                MSE_EKFs(m) = mean(MSE_aux);
                loglik(m) = loglik_aux(end);
                Px_init(:,:,m) = Px_est;

            end
        end
        
        % Estimates
        Delta = -loglik + min(loglik) - log(sum(W*exp(-loglik+min(loglik))));
        theta_est(:,t) = sum((W'.*exp(Delta)*ones(1,dim_theta))'.*tht_points,2);
        x_est(:,t) = sum((W'.*exp(Delta)*ones(1,dim_x))'.*Xpoints,2);
        
        Ptht_aux = 0;
        for m = 1:M
            Ptht_aux = Ptht_aux + W(m)*(tht_points(:,m)-theta_est(:,t))*(tht_points(:,m)-theta_est(:,t))'*exp(Delta(m));
        end
        Ptht_est(:,:,t) = Ptht_aux;
        
        NGF_NMSEx(t) = sum((x(:,t)-x_est(:,t)).^2)./sum((x(:,t)).^2);   
        NGF_NMSEtheta(t)=sum((A'-theta_est(:,t)).^2)./sum((A').^2);
        
        
 
        % Figures
        if figures_on == 1
            if mod(t,print_every)==1

                figure(4)
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
                semilogy(NGF_NMSEtheta(1:Tobs:t))
                ylabel('NMSE_\theta')
                subplot(2,3,6)
                semilogy(NGF_NMSEx(1:Tobs:t))
                ylabel('NMSE_x')
                
               
                sgtitle('NGF (UKF-EKF) - Lorenz 63 model')
                pause(0.1)

            end %mod(t,print_every)
               
        end % end figures   
        
        
    end % mod(t,Tobs)

end


% total time of simulation 
ttotal = etime(clock,t0)/60;

% Save data
clearvars -except NGF_NMSEtheta NGF_NMSEx x_est theta_est dim_y Tobs iter A x y H ttotal lambda
etiq_save = sprintf('data/NGF_UKFEKF_dimobs%d_iter%d.mat',dim_y, iter);
save(etiq_save);



end