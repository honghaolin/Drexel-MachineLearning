%% Support Vector Machines
%   Author: Haolin Hong
%   Date:   2017-Mar-2
%   Course: CS 383 - Assignment 7

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

%% Train a SVM on this training data and Classify/test SVM
% training
SVMModel = fitcsvm(data_training(:, 1:end-1), data_training(:, end));

% testing
predictVal = predict(SVMModel, data_testing(:, 1:end-1));

% clean temp variables
clear SVMModel;

%% Computes and Print Statistics
% count for Error Types
TP = 0;
FP = 0;
TN = 0;
FN = 0;
for i = 1 : length(predictVal)
    if predictVal(i) == 1
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

% compute statistics
precision = TP / (TP + FP);
recall = TP / (TP + FN);
fmeasure = 2 * precision * recall / (precision + recall);
accuracy = (TP + TN) / (TP + TN + FP + FN);

% print out results
fprintf('Precision: %f\n', precision);
fprintf('Recall: %f\n', recall);
fprintf('F-Measure: %f\n', fmeasure);
fprintf('Accuracy: %f\n', accuracy);

% clean temp variables
clear TP FP TN FN i;

%% Set environment back and clean
% retrieve the saving variables
load('env_backup.mat');

% remove backup file
delete('env_backup.mat');
