CS383 - HW1 - ReadMe
By Haolin Hong


To run the script, please open Matlab, switch to the folder contain both source.m file and diabetes.csv file. Then, type source in command window.


If it is not the first time running this script and you have diabetes.mat file in the same folder, it will load the .mat file automatically instead of loading from .csv file.
Thus, please make sure you do not have other file named diabetes.mat in the running folder (it may cause an error).

This script has been separated into some sections.
1. Clean up the environment
This section will back all the variable in working space into a .mat file called env_backup and clear the variables

2. Read in the data
This section will check if there exist the diabetes.mat file. If not, load from diabetes.csv file and save the result into diabetes.mat, otherwise, load the .mat file. 

3. Standardizes the data
This section will standardize the data by subtract the mean and divide their std.

4. Reduces data to 2D using PCA
First, set the value k. If k is greater than the total number of features, it will be set to the same with total features. Then, PCA reduce. Finally, project the data.

5. Graphs the data for visualization
This section will create a new figure and plot the projected data. Data plot will based on class label, and separated into red circle sign and blue cross sign.

6. Set environment back and clean
This section will load original data back and deleted the backup file.


All scripts has been tested. It suppose to work very well.
Thus, if there are any bug or error occurs, please email me.
Email: hh443@drexel.edu
