CS383 - HW2 - ReadMe
By Haolin Hong


To run the script, please open Matlab, switch to the folder contain both source.m file and diabetes.csv file. Then, type source in command window.


If it is not the first time running this script and you have diabetes.mat file in the same folder, it will load the .mat file automatically instead of loading from .csv file.
Thus, please make sure you do not have other file named diabetes.mat in the running folder (it may cause an error).


This script has been separated into some sections.
1. Clean up the environment
This section will back all the variable in working space into a .mat file called env_backup and clear the variables

2. Read in and Standardizes the data
This section will check if there exist the diabetes.mat file. If not, load from diabetes.csv file, pick the 6th and 7th feature only, then standardize them and save the result into diabetes.mat, otherwise, load the .mat file.

3. Set the k value
This section will set the k value. (How many cluster do we want)

4. Initial reference vectors
First, the random seed at 0. Compute the reference vectors randomly and plot the data.

5. Perform first itrations
This section is the first iteration for k-mean algorithm. It will class data to clusters, plot the result, and update the clusters.

6. Perform rest itrationes until end case
This section will loop until the sum of magnitude of change is less than eps. Then, plot the data.

7. Set environment back and clean
This section will load original data back and deleted the backup file.


All scripts has been tested. It suppose to work very well.
Thus, if there are any bug or error occurs, please email me.
Email: hh443@drexel.edu
