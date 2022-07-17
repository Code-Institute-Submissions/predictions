# numpy is used for array containers
import numpy as np
import pandas as pd 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Data Collection and Processing
# Loading the dataset to pandas DataFrame
loan_dataset = pd.read_csv('loan_dataset.csv')
print(type(loan_dataset))
print(loan_dataset.head())

# number of rows and columns
print(loan_dataset.shape)

# statistical measures
print(loan_dataset.describe())

# number of missing values in each column
print(loan_dataset.isnull().sum())

# dropping the missing values
loan_dataset = loan_dataset.dropna()
print(loan_dataset.isnull().sum())

# label encoding
loan_dataset.replace({"Loan_Status":{'N':0, 'Y':1}},inplace=True)
#print(loan_dataset)
print(loan_dataset.head())

# Dependent column values
print(loan_dataset['Dependents'].value_counts())

# Replacing the value of 3+ to 4
loan_dataset = loan_dataset.replace(to_replace='3+', value=4)

print(loan_dataset['Dependents'].value_counts())

# Data visualization, education & Loan Status
print(sns.countplot(x='Education', hue='Loan_Status', data=loan_dataset))
