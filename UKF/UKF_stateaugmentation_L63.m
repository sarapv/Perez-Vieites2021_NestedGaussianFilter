function [UKF_NMSEx,UKF_NMSEparam,x_est,theta_est,ttotal] = UKF_stateaugmentation_L63(x,y,s2x,s2y,h,T,Tobs,A,ko,H,dim_theta,iter,figures_on,print_every)
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


% Number of sigma points
M = 2*totaldim+1;
% Weights
ki = 1;
W = zeros(1,M);
W(1) =  ki/(totaldim+ki);
W(2:end) = 1/(2*(totaldim+ki));



% Init. variables
AXest = zeros(totaldim,T);      % estimations of x and theta
AXp = zeros(totaldim,M);         % augmented sigma-points (theta+x)
UKF_NMSEx = zeros(1,T);
UKF_NMSEparam = zeros(1,T);


% Initialization of theta and x
% theta
epsilon = [3 1 0.5]; 
Amu = A - epsilon + 2*epsilon.*rand(1,3);
AXest(1:dim_theta,1) = Amu' + randn(dim_theta,1);
Ptheta = 10*eye(totaldim); % cov. matrix of theta
% x
xinit = [-6; -5.5; 24.5] + randn(3,1);
AXest(dim_theta+1:end,1) = xinit;

% Sequential filtering
for t = 2:T
    
    % Every time there is a new observation (every Tobs steps)
    if mod(t,Tobs)==1
        
        %PREDICTION
        % Generate sigma-points
        S = ldl((totaldim+ki)*Ptheta);
        AXp = AXest(:,t-Tobs)*ones(1,M);
        AXp(:,2:(totaldim+1)) = AXp(:,2:(totaldim+1)) + S';
        AXp(:,(totaldim+2):end) = AXp(:,(totaldim+2):end) - S';
        
        % Propagate sigma-points
        propagate_x = zeros(dim_x,M);    
        propagate_x(1,:) = AXp(4,:) + h*( AXp(1,:).*(AXp(5,:)-AXp(4,:)) );% + sqrt(s2x)*randn(1,M);
        propagate_x(2,:) = AXp(5,:) + h*( AXp(4,:).*AXp(2,:) - AXp(4,:).*AXp(6,:) -AXp(5,:) );% + sqrt(s2x)*randn(1,M);
        propagate_x(3,:) = AXp(6,:) + h*( AXp(4,:).*AXp(5,:) - AXp(3,:).*AXp(6,:) );% + sqrt(s2x)*randn(1,M);
            
        Xpropagated = [AXp(1:dim_theta,:) ; propagate_x];
        
         % A priori state estimate and a priori covariance
        apriori_mean = sum((ones(totaldim,1)*W).*Xpropagated,2);
        apriori_cov = zeros(totaldim);
        for i=1:M
            aux = W(i)*(Xpropagated(:,i)-apriori_mean)*(Xpropagated(:,i)-apriori_mean)';
            apriori_cov = apriori_cov+aux;
        end
        apriori_cov = apriori_cov + [0*eye(dim_theta), zeros(dim_theta,dim_x); zeros(dim_x,dim_theta), s2x*eye(dim_x)];


        % UPDATE        
        Y = H*ko*Xpropagated(1+dim_theta:end,:);% + sqrt(s2y)*randn(dimobs,M);

        % Mean and covariace of the measurement vector
        obs_mean = sum((ones(dim_y,1)*W).*Y,2);
        obs_cov = zeros(dim_y);
        for i=1:M
            aux = W(i)*((Y(:,i)-obs_mean)*(Y(:,i)-obs_mean)');
            obs_cov = obs_cov+aux;
        end
        obs_cov = obs_cov + s2y*eye(dim_y);
        
        %Cross covariance
        cross_cov = zeros(totaldim,dim_y);
        for i=1:M
            aux  = W(i)*((Xpropagated(:,i)-apriori_mean)*(Y(:,i)-obs_mean)');
            cross_cov = cross_cov + aux;
        end
         
        % Kalman gain and update (mean and cov. theta)
        Kalman_gain = cross_cov*inv(obs_cov);
        AXest(:,t) = apriori_mean + Kalman_gain*(y(:,t)-obs_mean);
        Ptheta = apriori_cov - Kalman_gain*obs_cov*Kalman_gain';

        % NMSE
        UKF_NMSEparam(t) = sum((A'-AXest(1:dim_theta,t)).^2)./sum(A'.^2);
        UKF_NMSEx(t) = sum((x(:,t)-AXest(1+dim_theta:end,t)).^2)./sum(x(:,t).^2);
        
 
        % Figures
        if figures_on == 1
            if mod(t,print_every)==1

                figure(1)
                for j = 1:3
                    % Slow variables
                    subplot(2,3,j),
                    plot(1+Tobs:Tobs:t, AXest(3+j,1+Tobs:Tobs:t),'b'), hold on
                    plot(1+Tobs:Tobs:t, x(j,1+Tobs:Tobs:t),'k')
                    hold off

                end

                % Parameters
                subplot(2,3,4), 
                plot(1+Tobs:Tobs:t, AXest(1,1+Tobs:Tobs:t),'b'),
                hold on,         
                plot(1+Tobs:Tobs:t, A(1)*ones(size(1+Tobs:Tobs:t)),'b--'), 
                plot(1+Tobs:Tobs:t, AXest(2,1+Tobs:Tobs:t),'r'),
                plot(1+Tobs:Tobs:t, A(2)*ones(size(1+Tobs:Tobs:t)),'r--'), 
                plot(1+Tobs:Tobs:t, AXest(3,1+Tobs:Tobs:t),'g'),
                plot(1+Tobs:Tobs:t, A(3)*ones(size(1+Tobs:Tobs:t)),'g--'), 
                hold off

                % NMSE
                subplot(2,3,5)
                semilogy(UKF_NMSEparam(1:Tobs:t))
                ylabel('NMSE_\theta')
                subplot(2,3,6)
                semilogy(UKF_NMSEx(1:Tobs:t))
                ylabel('NMSE_x')
                
               
                sgtitle('UKF (state augm.) - Lorenz 63 model')
                pause(0.1)

            end %mod(t,print_every)
               
        end % end figures   
        
        
    end % mod(t,Tobs)

end

% Separate x and theta estiamtes in two variables
x_est = AXest(1+dim_theta:end,:);
theta_est = AXest(1:dim_theta,:);

% total time of simulation 
ttotal = etime(clock,t0)/60;

% Save data
clearvars -except UKF_NMSEparam UKF_NMSEx AXest dim_y Tobs x_est theta_est iter A x y H ttotal
etiq_save = sprintf('data/UKF_dimobs%d_iter%d.mat',dim_y, iter);
save(etiq_save);

end

