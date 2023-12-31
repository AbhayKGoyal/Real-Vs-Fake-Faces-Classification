function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


h=sigmoid(X*theta);
J=(1/m)*sum((-y.*log(h))-(1-y).*log(1-h))+(lambda/(2*m))*sum(theta(2:end).^2);


 for j=1:1:length(theta)
     sum1=0;
     if j==1
     for i=1:1:m
         h(i)=sigmoid(theta'*X(i,:)');
     sum1=sum1+(h(i)-y(i))*X(i,j);
     end
     grad(j)=sum1/m;
     
     else
          for i=1:1:m
         h(i)=sigmoid(theta'*X(i,:)');
     sum1=sum1+(h(i)-y(i))*X(i,j);
     end
     grad(j)=(sum1/m)+(lambda/m)*theta(j);
 end

% grad=(1/m)* X'*(h-y);



% =============================================================

end

