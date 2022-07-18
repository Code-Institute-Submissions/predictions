from enum import IntEnum
import pandas, os
from sklearn import svm

# Data loading
print("Welcome to lender assistant")
print("Loading history data")
loanDataset = pandas.read_csv('loan_dataset.csv')
print(loanDataset.head())

print("Preprocessing data")
# Drops rows with missing values from the dataset
loanDataset = loanDataset.dropna()

# using Intenum to replace Strings with integer values
class LoanStatus(IntEnum):
    REJECTED = 0
    ACCEPTED = 1
loanDataset.replace({"Loan_Status": {'N': LoanStatus.REJECTED.value, 'Y': LoanStatus.ACCEPTED.value}}, inplace=True)
print(loanDataset.head())
print(loanDataset['Loan_Status'].value_counts())

class Dependents(IntEnum):
    ZERO = 0
    ONE = 1
    TWO = 2
    MANY = 3
loanDataset.replace({"Dependents": {'3+': Dependents.MANY.value}}, inplace=True)
print(loanDataset.head())
print(loanDataset['Dependents'].value_counts())

class PropertyArea(IntEnum):
    URBAN = 0
    SEMI_URBAN = 1
    RURAL = 2
loanDataset.replace({"Property_Area": {'Urban': PropertyArea.RURAL.value}}, inplace=True)
print(loanDataset.head())
print(loanDataset['Property_Area'].value_counts())

class Gender(IntEnum):
    MALE = 0
    FEMALE = 1
loanDataset.replace({"Gender": {'Male': Gender.MALE.value, 'Female': Gender.FEMALE.value}}, inplace=True)
print(loanDataset.head())
print(loanDataset['Gender'].value_counts())

class Education(IntEnum):
    GRADUATE = 1
    NOT_GRADUATE = 0
loanDataset.replace({"Education": {'Graduate': Education.GRADUATE.value,'Not Graduate': Education.NOT_GRADUATE.value}}, inplace=True)
print(loanDataset.head())
print(loanDataset['Education'].value_counts())

class Married(IntEnum):
    SINGLE = 0
    MARRIED = 1
loanDataset.replace({"Married": {'No': Married.SINGLE.value, 'Yes': Married.SINGLE.value}}, inplace=True)
print(loanDataset.head())
print(loanDataset['Married'].value_counts())

class SelfEmployed(IntEnum):
    SELF_EMPLOYED = 0
    NOT_SELF_EMPLOYED = 1
loanDataset.replace({'Self_Employed': {'No': SelfEmployed.SELF_EMPLOYED.value, 'Yes': SelfEmployed.SELF_EMPLOYED.value}}, inplace=True)
print(loanDataset.head())
print(loanDataset['Self_Employed'].value_counts())

print("Training SVM, please wait")
trainingData = loanDataset.drop(columns=['Loan_ID', 'Loan_Status'], axis=1) #axis equal 1 removes the entire column
inferenceData = loanDataset['Loan_Status']
