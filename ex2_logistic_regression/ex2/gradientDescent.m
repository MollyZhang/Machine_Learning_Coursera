function [theta, J_history] = gradientDesecent(initial_theta, X, y, alpha, iter)

J_history = zeros(iter, 1);
theta = zeros(size(initial_theta));
for k=1:iter
    [J, gradient] = costFunction(theta, X, y);
    theta = theta - alpha * gradient;
    J_history(k) = J;
end

