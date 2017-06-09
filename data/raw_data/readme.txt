
The Fragile Families Challenge
Readme for data files
Version: 4 April 2017
*********************************

This zipped folder contains three comma-separated values files.

*********************************

background.csv contains 4,242 rows (one per child) and 12,944 columns:

challengeID: A unique numeric identifier for each child.

12,943 background variables asked from birth to age 9, which you may use in building your model.

*********************************

train.csv contains 2,121 rows (one per child in the training set) and 7 columns:

challengeID: A unique numeric identifier for each child.

Six outcome variables

Continuous variables: grit, gpa, materialHardship

Binary variables: eviction, layoff, jobTraining

*********************************

prediction.csv contains 4,242 rows and 7 columns:

This file is provided as a skeleton for your submission. You will submit a file in exactly this form but with your predictions for all 4,242 children included.

The file contains:

challengeID: A unique numeric identifier for each child.

Six outcome variables, as in train.csv. These are all filled with the mean value in the training set.

*********************************

Useful links:

Challenge website is http://www.fragilefamilieschallenge.org/

Submission platform is http://codalab.fragilefamilieschallenge.org/

Blog posts describing how to upload a contribution, describing variables, providing scientific motivation, etc. are at http://www.fragilefamilieschallenge.org/blog-posts/

Documentation for the background variables is at http://www.fragilefamilies.princeton.edu/

