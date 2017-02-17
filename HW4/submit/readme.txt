CS383 - HW4 - ReadMe
By Haolin Hong


To run the script, please open Matlab, switch to the folder contain both GradientDescent.m file and x06Simple.csv file. Then, type source in command window.
This script requires MATLAB® R2016b or later for in script functions.

If it is not the first time running this script and you have x06Simple.mat file in the same folder, it will load the .mat file automatically instead of loading from .csv file.
Thus, please make sure you do not have other file named diabetes.mat in the running folder (it may cause an error).


This script has been separated into some sections.
1. Clean up the environment
This section will back all the variable in working space into a .mat file called env_backup and clear the variables

2. Read in the data
This section will load data from x06Simple.csv or x06Simple.mat. If loading from csv file, it will ignoring the first row (header) and first column (index).

3. Get training data and testing data
This section will randomized observation first and select first 2/3(round up) data for training and remaining for testing.

4. Standardizes the data
In this section, the script will find the mean and standard deviation from training data, and then standardizes both training and testing data.

5. Batch Gradient Descent
Do the iterations by using Batch Gradient Descent and recode RMSE.
Initialize the parameters of theta using random values in the range [-1, 1]

6. Print and Plot
This section will display the final model, print final RMSE for testing data and plot RMSE for each iterations.

7. Set environment back and clean
This section will load original data back and deleted the backup file.

8. Function Implementation
This section required MATLAB® R2016b or later.
[BGD]
The batch gradient descent formular that used to update model
@(data) a matrix that row for observation and col for feature, last col for Y
@(weights) a one col matrix, first row for offset feature
@(eta) the learning rate
[RMSE]
Compute the root mean squared error for given data and weights
@(data) a matrix that row for observation and col for feature, last col for Y
@(weights) a one col matrix, first row for offset feature

All scripts has been tested. It suppose to work very well.
Thus, if there are any bug or error occurs, please email me.
Email: hh443@drexel.edu
