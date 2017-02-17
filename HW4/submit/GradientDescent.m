%% Gradient Descent
%   Author: Haolin Hong
%   Date:   2017-Feb-9
%   Course: CS 383 - Assignment 4

%% Clean up the environment
% save all variables from the workspace
save('env_backup.mat');

% clear all variables
clear variables;

%% Reads in the data
filename = 'x06Simple.csv';
datafile = 'x06Simple.mat';

if(exist(datafile, 'file'))
    % load data file if it exit
    load(datafile);
else
    % load data from csv file
    % ignoring the first row (header) and first column (index)
    data = csvread(filename, 1, 1);
    
    % save the data to datafile
    save(datafile,'data');
end

% clean temp variables
clear filename datafile;

%% Get training data and testing data
% randomizes the data
rng(0);
data = data( randperm( length(data) ), : );

% selects the first 2/3 (round up) of the data for training
num = ceil( length(data) * 2 / 3 );
data_training = data(1 : num, :);

% set the remaining for testing
data_testing = data(num+1 : end, :);

% clean temp variables
clear data num;

%% Standardizes the data
% find the mean and standard deviation of the training data
mv = mean(data_training(:, 1:end-1));
sd = std(data_training(:, 1:end-1));

% standardizes data
data_training = [(data_training(:, 1:end-1) - mv) ./ sd, data_training(:, end)];
data_testing = [(data_testing(:, 1:end-1) - mv) ./ sd, data_testing(:, end)];

% clean temp variables
clear mv sd;

%% Batch Gradient Descent
% Declare the max iteration number and learning rate
iteration_max = 1000000;
eta = 0.01; % learning rate

% Initialize the parameters of theta using random values in the range [-1, 1]
theta = rand(size(data_training, 2), 1) * 2 - 1; % each col is a weights for iteration

% Create variable to save RMSEs
rmse_training = [0]; %#ok<NBRAK>
rmse_testing = [0]; %#ok<NBRAK>

% Iteration
for i = 1 : iteration_max
    theta(:, end+1) = BGD(data_training, theta(:, end), eta); %#ok<SAGROW>
    rmse_training(end+1) = RMSE(data_training, theta(:, end)); %#ok<SAGROW>
    rmse_testing(end+1) = RMSE(data_testing, theta(:, end)); %#ok<SAGROW>
    if (abs(rmse_training(end) - rmse_training(end-1)) < eps && i ~= 1)
        break;
    end
end

% Get the final model
weights = theta(:, end);

% clean temp variables
clear iteration_max eta theta i;

%% Print and Plot
% Display final model
display(weights);

% Print RMSE
fprintf('The final RMSE testing error is %f\n', rmse_testing(end));

% Remove the first value (which is zero)
rmse_training = rmse_training(2:end);
rmse_testing = rmse_testing(2:end);

% Plotting of RMSE for each iteration on a new figure
figure;
hold on;
plot(rmse_training, 'r-');
plot(rmse_testing, 'b-');
hold off;

% Set legend and labels
legend('Training Error','Testing Error');
xlabel('Iteration');
ylabel('RMSE of Training and Testing Data');

% clean temp variables
clear rmse_training rmse_testing;

%% Set environment back and clean
% retrieve the saving variables
load('env_backup.mat');

% remove backup file
delete('env_backup.mat');

%% Function Implementation
% The batch gradient descent formula that used to update model
% @(data) a matrix that row for observation and col for feature, last col for Y
% @(weights) a one col matrix, first row for offset feature
% @(eta) the learning rate
function theta = BGD(data, weights, eta)
    theta = zeros(length(weights), 1);
    X = [ones(length(data) ,1), data(:, 1:end-1)];
    Y = data(:, end);
    for j = 1 : length(theta)
        s = 0;
        for i = 1 : length(data)
            s = s + (X(i, :) * weights - Y(i)) * X(i,j);
        end
        theta(j) = weights(j) - (eta/length(data)) * s;
    end
end

% Compute the root mean squared error for given data and weights
% @(data) a matrix that row for observation and col for feature, last col for Y
% @(weights) a one col matrix, first row for offset feature
function rmse = RMSE(data, weights)
    X = [ones(length(data), 1), data(:, 1:end-1)];
    Y = data(:, end);
    predict = X * weights;
    rmse = sqrt(mean( (predict - Y).^2 ));
end
