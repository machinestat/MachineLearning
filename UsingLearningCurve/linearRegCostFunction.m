function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
% LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
% regression with multiple variables. Returns the cost in J and the 
% gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% return the following variables
J = 0;
grad = zeros(size(theta));

% Compute the cost and gradient of regularized linear regression for 
% a particular choice of theta.

J = (1/(2*m))*(X*theta - y)' * (X*theta - y);
grad = (1/m)*(X' * (X*theta - y));

theta(1) = 0;
sstheta = sum(theta .^2);
J = J + (lambda/(2*m))*sstheta;
grad = grad + (lambda/m)*theta;

% =========================================================================

grad = grad(:);

end
