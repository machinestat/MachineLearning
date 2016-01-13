% use this Matlab/Octave code for fitting linear regression models
% using gradient descent

% Part I: Linear regression with one varible
% load the data set
data = load('Data/ex1data1.txt');
X = data(:, 1);
y = data(:, 2);
m = length(y); % number of training example
% Plotting the data
plotData(X, y);

% Gradient Descent
X = [ones(m, 1), data(:, 1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

% Some gradient descent settings
iterations = 1500;
alpha = 0.01;

% compute and display initial cost
computeCost(X, y, theta)

% run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations);

% print theta to screen
fprintf('Theta found by gradient descent: ');
fprintf('%f %f \n', theta(1), theta(2));

% plot the linear fit
hold on; % keep previous plot visible
plot(X(:, 2), X*theta, "-")
legend('Training data', 'Linear regression')
hold off; % no more plots on this figure

% Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] *theta;
fprintf('For population = 35,000, we predict a profit of %f\n',...
    predict1*10000);
predict2 = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n',...
    predict2*10000);

% Visualizing the cost funtion 

% Grid over calculating J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];    
	  J_vals(i,j) = computeCost(X, y, t);
    end
end

% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);

% Part II: Linear regression with multiple variables
%% Clear and Close Figures
clear ; close all; clc
%% Load Data
data = load('Data/ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

% Scale features and set them to zero mean
[X mu sigma] = featureNormalize(X);
% Add intercept term to X
X = [ones(m, 1) X];

% Gradient Descent 
% Choose some alpha value
alpha = 0.01;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

%----------------------------------Functions---------------------------------
function plotData(x, y)
% function plotData plots the data points in x and y into a new figure using 
% the "figure" and "plot" commands. Set the axes labels  using "xlabel" and 
% "ylabel" commands. The "rx" option set the markers as red crosses

figure; % open a new figure window
plot(x, y, 'rx', 'MarkerSize', 10); % plot the data
ylabel('Profit in $10, 000s'); % set the y-axis label
xlabel('Population of City in 10, 000s'); % set the x-axis label

end

function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression

% Initialize some useful values
m = length(y); % number of training examples
J = 0;

h = X * theta; % first compute the hypothesis values
error = h - y; % compute the difference between h and y
% compute the square of each of those error terms using element-wise 
% exponentiation
error_sqr = error .^ 2;
J = (1/(2*m))*(sum(error_sqr)); % calculate the cost
end

function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
% function GRADIENTDESCENT Performs gradient descent to learn theta

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

  for iter = 1:num_iters
    h = X*theta - y;
    theta_change = alpha*(1/m)*(X' * h);
    theta = theta - theta_change;
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
  end
end
  
function [X_norm, mu, sigma] = featureNormalize(X)
%This function normalizes the features in X 
%FEATURENORMALIZE(X) returns a normalized version of X where
%the mean value of each feature is 0 and the standard deviation
%is 1. 

X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));
m = size(X, 1);

mu = mean(X); %compute the mean of each column in X
sigma = std(X); % compute the standard deviation for each variable in X
mu_matrix = ones(m, 1) * mu;
sigma_matrix = ones(m, 1) * sigma;
X_norm = (X - mu_matrix) ./ sigma_matrix;
end