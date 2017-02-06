%% Homework 2
%   Author: Haolin Hong
%   Date:   2017-Jan-26
%   Course: CS 383

%  using just the 6th and 7th feature of the data with k=2
%  and using the Euclidean (L2) distance.

%% Clean up the enviroment
% save all variables from the workspace
save('env_backup.mat');

% clear all variables
clear variables;

%% Reads in and Standrdizes the data
filename = 'diabetes.csv';
datafile = 'diabetes.mat';

if(exist(datafile, 'file'))
    % load data file if it exit
    load(datafile);
else
    % load data from csv file
    data = csvread(filename);
    
    % select 6th and 7th feature of data
    data = data(:, 7:8);
    
    % standarizes the data
    data = (data - mean(data)) ./ std(data);
    
    % save the data to datafile
    save(datafile,'data');
end

% clean temp variables
clear filename datafile;

%% Set the k value
k = 2;

%% Initial reference vectors
% seed the random number generator with zero and get random numbers
rng(0);
permutation = randperm(length(data), k);

% pick out the reference vectors
cluster = zeros(size(data, 2), k);
for i = 1 : k
    cluster(i,:) = data( permutation(i), : ) ;
end

% graphs the data for visualization
figure;
plot(...
    data(:, 2), data(:, 1), 'rx', ...
    cluster(:, 2), cluster(:, 1), 'bo'...
    );
title('Initial Seeds');

% clean temp variables
clear permutation i;

%% Perform first itrations
% compute euclidean distance of each observation
distance = zeros(length(data), k);
for i = 1 : size(distance, 1)
    observation = data(i, :);
    for j = 1 : size(distance, 2)
        distance(i, j) = sqrt(sum( (cluster(j, :) - observation).^2 ));
    end
end

% classify data by distance
[~, class] = min(distance, [], 2);

% graphs the data for visualization
figure;
plot(...
    data(class == 1, 2), data(class == 1, 1), 'rx', ...
    cluster(1, 2), cluster(1, 1), 'ro', ...
    data(class == 2, 2), data(class == 2, 1), 'bx', ...
    cluster(2, 2), cluster(2, 1), 'bo'...
    );
title('Iteration 1');

% compute new reference vector and track the magnitude change
magnitude_change = 0;
for i = 1 : k
    avg = mean(data(class == i, :));
    magnitude_change = sqrt(sum( (cluster(i, :) - avg).^2 ));
    cluster(i, :) = avg;
end

% clean temp variables
clear distance observation i j;

%% Perform rest itrationes until end case
% initial iteration counter, start with 1
iteration = 1;

% terminate the EM process until the sum of magnitude of change is less than eps
while(magnitude_change >= eps)
    % compute euclidean distance of each observation
    distance = zeros(length(data), k);
    for i = 1 : size(distance, 1)
        observation = data(i, :);
        for j = 1 : size(distance, 2)
            distance(i, j) = sqrt(sum( (cluster(j, :) - observation).^2 ));
        end
    end
    
    % classify them by distance
    [~, class] = min(distance, [], 2);
    
    % compute new reference vector and track the magnitude change
    magnitude_change = 0;
    for i = 1 : k
        avg = mean(data(class == i, :));
        magnitude_change = sqrt(sum( (cluster(i, :) - avg).^2 ));
        cluster(i, :) = avg;
    end
    
    iteration = iteration + 1;
end

% graphs the data for visualization
figure;
plot(...
    data(class == 1, 2), data(class == 1, 1), 'rx', ...
    cluster(1, 2), cluster(1, 1), 'ro', ...
    data(class == 2, 2), data(class == 2, 1), 'bx', ...
    cluster(2, 2), cluster(2, 1), 'bo'...
    );
title('Iteration ' + string(iteration));

% clean temp variables
clear distance observation magnitude_change i j;

%% Set environment back and clean
% retrieve the saving variables
load('env_backup.mat');

% remove backup file
delete('env_backup.mat');
