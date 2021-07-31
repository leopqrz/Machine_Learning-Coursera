function [cost, theta] = costFunction(X, y, theta)

  m = length(y); % number of training examples
  h = X*theta;
  cost = (-1/(2*m)) * (X*theta - y)' * (X*theta - y);
  theta = (1/m) * X'*(h-y);
  
end