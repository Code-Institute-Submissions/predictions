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

# marital status & Loan Status
sns.countplot(x='Married', hue='Loan_Status', data=loan_dataset)

# Convert to numerical values 
loan_dataset.replace({'Married':{'N':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
                    'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)
print(loan_dataset.head())

# separating the data and label
X = loan_dataset.drop(columns=['Loan_ID', 'Loan_Status'],axis=1)
Y = loan_dataset['Loan_Status']
print(X)
print(Y)

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)
#print(X.shape,X_train.shape,X_test.shape)

# SVM Model
classifier = svm.SVC(kernel='linear')
# training the SVM model
classifier.fit(X_train,Y_train)

# Model Evaluation, accuracy score on training data
#X_train_prediction = classifier.predict(X_train)
#training_data_accuracy = accuracy_score(X_train_prediction,Y_train)

#print('Accuracy on training data: ', training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
training_data_accuracy = accuracy_score(X_test_prediction,Y_test)

print('Accuracy on test data: ', training_data_accuracy)