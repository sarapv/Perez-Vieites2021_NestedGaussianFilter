function [x_est,MSEx,loglik_aux,Px_upd] = EKFt0_tL63(x,y,xinit,Pinit,t0,T,Tobs,s2x,s2y,H,ko,h,A)
% Extended Kalman filter for the Lorenz 63 model. It can run from time t0
% to time T


dim_x = size(x,1);
dim_y = size(H,1);
loglik_aux = 0;
J = zeros(dim_x,dim_x);

% init
x_est = zeros(dim_x,T);
x_est(:,1) = xinit;
x_aux = x_est(:,1);
Px_aux = Pinit;


for t = t0+1:T
   
    % Prediction (mean)
    xpred = zeros(3,1);
    xpred(1) = x_aux(1) + h*( A(1)*(x_aux(2)-x_aux(1)) );
    xpred(2) = x_aux(2) + h*( x_aux(1)*A(2) - x_aux(1)*x_aux(3) -x_aux(2) );
    xpred(3) = x_aux(3) + h*( x_aux(1)*x_aux(2) - A(3)*x_aux(3) );
   
    %Compute Jacobian  
    J(1,1)=1-(A(1)*h);
    J(1,2)=A(1)*h;
    J(2,1)=h*(A(2)-x_aux(3));
    J(2,2)=1-h;
    J(2,3)= - h*x_aux(1);
    J(3,1)=h*x_aux(2);
    J(3,2)=h*x_aux(1);
    J(3,3)=1-(A(3)*h);
    
    % Prediction (cov)
    Px_pred = J*Px_aux*J' + s2x*eye(dim_x);
    
    
    % Update
    if mod(t,Tobs) == 1
      

       S = H*Px_pred*H' + s2y*eye(dim_y);
       Kalman_gain = (Px_pred*H')/S;
       innov = y(:,t) - H*(ko*xpred);
       x_est(:,t) = xpred + Kalman_gain*innov;
       Px_upd = (eye(dim_x) - Kalman_gain*H)*Px_pred;
       
       x_aux = x_est(:,t);
       Px_aux = Px_upd;
       
       
       loglik_aux =  [loglik_aux, (0.5*dim_x)*log(2*pi) + 0.5*log(det(S)) + 0.5*(innov'/S)*innov];
    
    else
       x_aux = xpred;
       Px_aux = Px_pred;
       
    end
    
    
end

MSEx = sum((x_est(:,1:Tobs:T) - x(:,1:Tobs:T)).^2)/dim_x;


end

