%% Closed Form Linear Regression
%   Author: Haolin Hong
%   Date:   2017-Feb-2
%   Course: CS 383 - Assignment 3

%% Clean up the enviroment
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

%% Computes the closed-form solution of linear regression
% split training data into x and y
x = data_training(:, 1:end-1);
y = data_training(:, end);

% add an addition feature with value 1 to the data
offset = ones(size(data_training, 1), 1);
x = [offset x];

% compute weights and print out
weights = (x' * x) \ x' * y; % theta = (X' * X )^(-1) * X' * Y
display(weights);

% clean temp variables
clear x y offset;

%% Computes the root mean squared error (RMSE)
% applies the solution to the testing samples
testing = [ones( size(data_testing, 1), 1 ) data_testing(:, 1:end-1)];
predict = testing * weights;

% compute root mean squared error
mse = mean( (predict - data_testing(:, end)).^2 );
rmse = sqrt(mse);

% print out the rmse
fprintf('RMSE for Closed Form Linear Regression is %f\n', rmse);

% clean temp vairbales
clear testing predict mse;

%% Set environment back and clean
% retrieve the saving variables
load('env_backup.mat');

% remove backup file
delete('env_backup.mat');
