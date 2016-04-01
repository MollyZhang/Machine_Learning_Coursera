function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
n = size(theta, 1); % number of features

% ====================== YOUR CODE HERE ======================

h = sigmoid(X * theta);

J = sum((-y.*log(h))-((1-y).*log(1-h)))/m + lambda/(2*m)*sum(theta(2:end).^2);

grad(1) = sum((h-y).* X(:,1))/m;
for k=2:n
    grad(k) = sum((h-y).* X(:,k))/m + lambda/m*theta(k);
end


% =============================================================

grad = grad(:);

end
