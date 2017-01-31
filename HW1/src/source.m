%% Homework 1
%   Author: Haolin Hong
%   Date:   2017-Jan-12
%   Course: CS 383

%% Clean up the enviroment
% save all variables from the workspace
save('env_backup.mat');

% clear all variables
clear variables;

%% Reads in the data
filename = 'diabetes.csv';
datafile = 'diabetes.mat';

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

%% Standardizes the data
% split class from data
class = data(:, 1);
data(:, 1) = [];

% compute average for each feature
avg = mean(data);

% compute standard deviation for each feature
sd = std(data);

% standarizes
stand = (data - avg) ./ sd;

% clean temp variables
clear avg sd;

%% Reduces data to 2D using PCA
% set the demitions
k = 2;
k = min(k, length(stand));

% compute and sort the eigenvalue and eigenvector
covariance = cov(stand);
[V, D] = eig(covariance);
[~, I] = sort(diag(D), 'descend');

% reduce using PCA
projected = zeros(size(stand, 1), k);
for i = 1: k
    projected(:, i) = stand * V(:, I(i));
end

% clean temp variables
clear covariance V D I i;

%% Graphs the data for visualization
% create new figure
figure;

% plot data using class label
plot(...
    projected(class == 1, 1), projected(class == 1, 2), 'ro', ...
    projected(class == -1, 1), projected(class == -1, 2), 'bx'...
);

% set labels
title('PCA');

%% Set environment back and clean
% retrieve the saving variables
load('env_backup.mat');

% remove backup file
delete('env_backup.mat');
