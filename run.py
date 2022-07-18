from enum import IntEnum
import pandas, os
from sklearn import svm

# Data loading
print("Welcome to lender assistant")
print("Loading history data")
loadDataset = pandas.read_csv('loan_dataset.csv')
print(loadDataset.head())