CS383 - HW5&6 - ReadMe
By Haolin Hong


To run the script, please open Matlab, switch to the folder contain both KNN.m file and spambase.data file. Then, type KNN in command window.
This script requires MATLAB® R2016b or later for in script functions.

If it is not the first time running this script and you have spambase.mat file in the same folder, it will load the .mat file automatically instead of loading from .data file.
Thus, please make sure you do not have other file named spambase.mat in the running folder (it may cause an error).


This script has been separated into some sections.
1. Clean up the environment
This section will back all the variable in working space into a .mat file called env_backup and clear the variables

2. Read in the data
This section will load data from spambase.data or spambase.mat. If loading from data file, it will ignoring the first row (header) and first column (index).

3. Get training data and testing data
This section will randomized observation first and select first 2/3(round up) data for training and remaining for testing.

4. Standardizes the data
In this section, the script will find the mean and standard deviation from training data, and then standardizes both training and testing data.

5. Predict for testing data
For each testing data, it will compute the similarity to each training data.
Then pick out the highest k of them to do prediction.

6. Computes and Print Statistics
This section will compute for precision, recall, f-measure and accuracy.
Then print them.

7. Set environment back and clean
This section will load original data back and deleted the backup file.

8. Function Implementation
This section required MATLAB® R2016b or later.
[BGD]
Compute the similarity for two given data, using Manhattan Distance.
@(data_1) matrix that has one row, and multi col for features.
@(data_2) matrix that has one row, and multi col for features.


All scripts has been tested. It suppose to work very well.
Thus, if there are any bug or error occurs, please email me.
Email: hh443@drexel.edu
