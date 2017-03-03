%% k-Nearest Neighbors
%   Author: Haolin Hong
%   Date:   2017-Mar-2
%   Course: CS 383 - Assignment 5&6

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

%% Performs k-Nearest Neighbors classification
% Declare the k
k = 5;

% Declare the predict matrix
predictVal = zeros(size(data_testing, 1));

% For each testing data
for i = 1 : size(data_testing, 1)
    testing = data_testing(i, :);

    % Compute similarity to each training data
    similarity = zeros(size(data_training, 1), 1);
    for j = 1 : size(data_training, 1)
        similarity(j) = kernel(testing(1:end-1), data_training(j, 1:end-1));
    end

    % Sort the similarity, and find k data which has lowest similarity
    [~, I] = sort(similarity);

    % Count for spam and not-spam
    spam = 0;
    for j = 1 : k
        if data_training(I(j), end) == 1
            spam = spam + 1;
        end
    end

    % Find out predict value
    if 2*spam > k
        predictVal(i) = 1;
    end
end

% clean temp variables
clear k i j testing similarity I spam;

%% Computes and Print Statistics
% Count for Error Types
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

% Compute statistics
precision = TP / (TP + FP);
recall = TP / (TP + FN);
fmeasure = 2 * precision * recall / (precision + recall);
accuracy = (TP + TN) / (TP + TN + FP + FN);

% Print out results
fprintf('Precision: %f\n', precision);
fprintf('Recall: %f\n', recall);
fprintf('F-Measure: %f\n', fmeasure);
fprintf('Accuracy: %f\n', accuracy);

% clean temp variables
clear TP FP TN FN;

%% Set environment back and clean
% retrieve the saving variables
load('env_backup.mat');

% remove backup file
delete('env_backup.mat');

%% Functions
% Compute the similarity for two given data, using Manhattan Distance.
% @(data_1) matrix that has one row, and multi col for features.
% @(data_2) matrix that has one row, and multi col for features.
function similarity = kernel(data_1, data_2)
% Return -1 if data_1 & data_2 do not have same length
if length(data_1) ~= length(data_2)
    similarity = -1;
    return
else
    len = length(data_1);
end
% Compute Manhattan Distance for data_1 & data_2
similarity = 0;
for i = 1 : len
    similarity = similarity + abs(data_1(i) - data_2(i));
end
end
