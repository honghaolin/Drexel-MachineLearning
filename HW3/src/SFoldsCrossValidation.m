%% S-Folds Cross-Validation
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

%% Randomizes the data and compute the length of S folds
% set S for S-Folds Cross-Validation
S = 5;

% randomizes the data
rng(0);
data = data( randperm( length(data) ), : );

% compute length s folds
len = ceil(length(data) / S);

%% Working on S flods
% Track the squared error
squaredError = [];

for i = 1 : S
    % Select fold i as testing data and the remaining folds as training data
    head = 1 + (i - 1) * len;
    tail = min(head + len - 1, length(data));
    data_testing = data(head:tail, :);
    data_training = [data(1:head-1, :); data(tail+1:end, :)];
    
    % Standardizes the data based on the training data(except for the last column)
    mv = mean(data_training(:, 1:end-1));
    sd = std(data_training(:, 1:end-1));
    data_training = [(data_training(:, 1:end-1) - mv) ./ sd, data_training(:, end)];
    data_testing = [(data_testing(:, 1:end-1) - mv) ./ sd, data_testing(:, end)];
    
    % Train a closed-form linear regression model
    x = [ones(size(data_training, 1), 1) data_training(:, 1:end-1)];
    y = data_training(:, end); % split training data into x and y
    weights = (x' * x) \ x' * y; % compute weights
    
    % Compute the squared error for each sample in the current testing fold
    testing = [ones( size(data_testing, 1), 1 ) data_testing(:, 1:end-1)];
    predict = testing * weights;
    squaredError = [squaredError; (predict - data_testing(:, end)).^2];
end

% clean temp variables
clear mv sd len i head tail data_testing data_training x y weights testing predict;

%% Compute the RMSE using all the errors
% compute root mean squared error and print it out
rmse = sqrt( mean( squaredError ) );
fprintf('RMSE for S-Fold Cross-Validation is %f\n', rmse);

% clean temp variables
clear squaredError;

%% Set environment back and clean
% retrieve the saving variables
load('env_backup.mat');

% remove backup file
delete('env_backup.mat');
