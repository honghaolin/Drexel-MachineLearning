%% Muti-Class Support Vector Machines
%   Author: Haolin Hong
%   Date:   2017-Mar-6
%   Course: CS 383 - Assignment 7

%% Clean up the environment
% save all variables from the workspace
save('env_backup.mat');

% clear all variables
clear variables;

%% Reads in the data
filename = 'CTG.csv';
datafile = 'CTG.mat';

if(exist(datafile, 'file'))
    % load data file if it exit
    load(datafile);
else
    % load data from csv file
    data = csvread(filename, 2, 0);
    data(:, end-1) = [];
    
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

%% Get Relevant Classes
% find total number of classes training data contain
index_max = max(data_training(:, end));

% saprate data based on class
class = cell(index_max, 1);
for i = 1 : index_max
    class{i} = data_training( data_training(:, end) == i , :);
end

% clean temp variables
clear index_max i;

%% Model Training
% train SVM models and save it into variable model
model = cell(0, 1);
for i = 1 : length(class) - 1
    for j = i + 1 : length(class)
        data = [class{i}; class{j}];
        model{end + 1} = fitcsvm(data(:, 1:end-1), data(:, end)); %#ok<SAGROW>
    end
end

% clean temp variables
clear i j data;

%% Comput Predict Values
% for each SVM models, predict result
pValues = zeros( size(data_testing, 1), length(model) );
for i = 1 : length(model)
    pValues(:, i) = predict(model{i}, data_testing(:, 1:end-1));
end

% find the class beat other most 
predictValue = mode(pValues, 2);

% clean temp variables
clear pValues i;

%% Compute Accuracy and Print out
% comput percentage of samples classified correctly confident.
correct = 0;
for i = 1 : length(predictValue)
    if (predictValue(i) == data_testing(i, end))
        correct = correct + 1;
    end
end
accuracy = correct / length(predictValue);

% print accuracy
fprintf('Accuracy: %f\n', accuracy);

% clean temp variables
clear correct i;

%% Set environment back and clean
% retrieve the saving variables
load('env_backup.mat');

% remove backup file
delete('env_backup.mat');
