# AI Lender Assistant

Lender Assistant is a AI Python program which use SVM to solve ML classification problems.
This program runs on the Code Institute mock terminal on Heroku.
The program show how an AI powered app can automate loan approval process without manual interventions.
The dataset from Kaggle contains parameterized data that influence the loan approval process.
This parameterized dataset is then used to train the classifier which is then used to predict whether the loan is to be approved or not.

## Features 

### Existing Features

- __Random generation of loan values__

  - Accepts user input.
  - Predicting on test dataset
  - Building ML learning model
  - Train the system
  - Importing dataset from CSV files
  - Pre-processing dataset for training
  - Creating SVM classifier
  - Training the SVM classifier with the preprocessed dataset.
  - Make predictions based on the trained model
 
![image](https://user-images.githubusercontent.com/101147217/180053041-325d30ae-7ca8-4664-8616-4e3f467d28e1.png)
![Eligible](https://user-images.githubusercontent.com/101147217/180053641-f0f486ab-8c38-49b9-ba67-3ac17ed3e6ea.png)
![Not Eligible](https://user-images.githubusercontent.com/101147217/180056875-cf1d0531-2a37-4de4-951f-0fa55e6b49de.png)

- __Future Features__

  - Utilise Google Sheets to store data
  - Analysis of data output to a graphical view
  - Add accuracy functionality
  - Add more data sets
  - Saving the trained classifier model to sav file
  - Loading pre-trained model from disk
  - Dataset splitting for accuracy measurement and analysis of the same
  - GUI for interactive user input and validation
  - API to interact with the AI program.
  - Normalization of parameterized data.


- __Data Model__

  - The program is trained to recognise a set of data and the algorithm will learn from those data
  - It will validate the relationship between the variables defined in the CSV file that has 614 rows and 13 columns

- __Testing__

  - Code was validated in PEP8 
  - Checked valid inputs 
  - Tested in Gitpod and the Code Institute Heroku terminal
  

- __Bugs__ 

	- __Solved Bugs__
  - Formatted strings values had incorrect values
  

- __Remaining Bugs__

  - No bugs remaining 


- __Validator Testing__

  - PEP8
	- A number of errors were returned, I corrected a number of them but some of them lines are above the 79 characters limit


### Deployment

- This project was deployed using Code Institute's mock terminal for Heroku
	- Steps
		- Fork or clone this repository
		- Create a new Heroku app
		- Set the buildbacks to Python
		- Link Heroku app to the repository
		- Click on Deploy

## Credits
	- Code Institute for the deployment terminal
	- (https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
	- (https://www.analyticsvidhya.com/blog/2022/02/loan-approval-prediction-machine-learning/)
	- (https://www.kaggle.com/datasets/ninzaami/loan-predication)
	- (https://www.geeksforgeeks.org/enum-intenum-in-python/)
  