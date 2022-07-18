from enum import IntEnum
import pandas, os
from sklearn import svm

# Data loading
print("Welcome to lender assistant")
print("Loading history data")
loadDataset = pandas.read_csv('loan_dataset.csv')
print(loadDataset.head())

print("Preprocessing data")
# Drops rows with missing values from the dataset
loadDataset = loadDataset.dropna()

# using Intenum to replace Strings with integer values
class LoanStatus(IntEnum):
    REJECTED = 0
    ACCEPTED = 1
loadDataset.replace({"Loan_Status": {'N': LoanStatus.REJECTED.value, 'Y': LoanStatus.ACCEPTED.value}}, inplace=True)
print(loadDataset.head())
print(loadDataset['Loan_Status'].value_counts())
