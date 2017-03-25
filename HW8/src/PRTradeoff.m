%% Precision Recall Trade off
%   Author: Haolin Hong
%   Date:   2017-Mar-25
%   Course: CS 383 - Assignment 8

%% Clean up the environment
% save all variables from the workspace
save('env_backup.mat');

% clear all variables
clear variables;

%% Reads in the data
filename = 'spambase.data';
datafile = 'spambase.mat';

if(exist(datafile, 'file'))
    % load data file if it exit
    load(datafile);
else
    % load data from csv file
    data = csvread(filename);
    
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

%% Trains an artificial neural network
% settings for traning model
eta = 0.5;
hidden_layer_size = 20;
iteration_max = 1000;

% size
size_hidden = hidden_layer_size;
size_input = size(data_training, 2);
size_output = 1;

N = length(data_training);

% initial weights
beta = rand(size_input, size_hidden) * 2 - 1;
theta = rand(size_hidden, size_output) * 2 - 1;

% data
data = [ones(N, 1), data_training(:, 1:end-1)];

% iterations
for i = 1 : iteration_max
    % forward propagation
    hidden = 1 ./ ( 1 + exp(-1 .* data * beta) );
    output = 1 ./ ( 1 + exp(-1 .* hidden * theta) );
    
    % back propagation
    delta_out = data_training(:, end) - output;
    theta = theta + (eta/N) .* (hidden' * delta_out);
    delta_hid = delta_out * theta' .* hidden .* (1 - hidden);
    beta = beta + (eta/N) .* (data' * delta_hid);
end

% clean temp variables
clear eta hidden_layer_size iteration_max ...
    size_hidden size_input size_output N ...
    data i hidden output delta_out delta_hid;

%% Classifies the testing data and Computes the testing error
% apply model
data = [ones(length(data_testing), 1), data_testing(:, 1:end-1)];
hidden = 1 ./ ( 1 + exp(-1 .* data * beta) );
output = 1 ./ ( 1 + exp(-1 .* hidden * theta) );

% inital percision and recall
precision = [];
recall = [];

for threshold = 0 : 0.1 : 1
    % classifies testing data
    predictValue = output;
    predictValue(predictValue < threshold) = 0;
    predictValue(predictValue >= threshold) = 1;
    
    % compute testing error
    TP = 0;
    FP = 0;
    TN = 0;
    FN = 0;
    
    for i = 1 : length(predictValue)
        if predictValue(i) == 1
            if data_testing(i, end) == 1
                TP = TP + 1;
            else
                FP = FP + 1;
            end
        else
            if data_testing(i, end) == 0
                TN = TN + 1;
            else
                FN = FN + 1;
            end
        end
    end
    
    if TP + FP == 0
        precision(end+1) = 1; %#ok<SAGROW>
    else
        precision(end+1) = TP / (TP + FP); %#ok<SAGROW>
    end
    if TP + FN == 0
        recall(end+1) = 0; %#ok<SAGROW>
    else
        recall(end+1) = TP / (TP + FN); %#ok<SAGROW>
    end
end

% plot precision vs recall
figure;
plot(precision, recall, 'o-');
title('PR-Graph for ANN');
xlabel('Precision');
ylabel('Recall');

% clean temp variables
clear threshold data hidden output predictValue TP FP TN FN i;

%% Set environment back and clean
% retrieve the saving variables
load('env_backup.mat');

% remove backup file
delete('env_backup.mat');
