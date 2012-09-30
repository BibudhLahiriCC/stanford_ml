function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
X = [ones(m, 1) X];
a_1 = X';
z_2 = Theta1*a_1;
a_2 = sigmoid(z_2);
a_2 = [ones(1, m); a_2];
z_3 = Theta2*a_2;
a_3 = sigmoid(z_3);
%a_3 will be a 10 x 5000 matrix. a_3 = Set of 5000 values of (h_{\theta}(x))

%Original y is an m x 1 vectors, where the elements are from {1,2,...K}.
%We convert it to an m x K matrix, where the i-th row and k-th col is 1
%y_i = k.
y_original = y;
y = zeros(m, num_labels);
for i = 1:m
  for j = 1:num_labels
    if (y_original(i) == j)
       y(i, j) = 1;
    else
       y(i, j) = 0;
    endif
  endfor
endfor

sum = 0;
for i = 1:m
  for k = 1:num_labels
    sum += (y(i, k)*log(a_3(k, i)) + (1 - y(i, k))*log(1 - a_3(k, i)));
  endfor
endfor
%disp(sum);
J = -(1/m)*sum;
%J = (-1/m)*(ones(1, num_labels)*(log(a_3)*y)*ones(num_labels, 1) +
%        + ones(1,m)*(ones(m, num_labels) - y)*log(ones(num_labels, m) - a_3)*ones(m,1));

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
