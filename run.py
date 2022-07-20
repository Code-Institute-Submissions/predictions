from enum import IntEnum
import pandas, os
from sklearn import svm

# Data loading
print("Welcome to lender assistant")
print("Loading history data")
loanDataset = pandas.read_csv('loan_dataset.csv')
#print(loanDataset.head())

print("Preprocessing data")
# Drops rows with missing values from the dataset
loanDataset = loanDataset.dropna()

# using Intenum to replace Strings with integer values
class LoanStatus(IntEnum):
    REJECTED = 0
    ACCEPTED = 1
loanDataset.replace({"Loan_Status": {'N': LoanStatus.REJECTED.value, 'Y': LoanStatus.ACCEPTED.value}}, inplace=True)
#print(loanDataset.head())
#print(loanDataset['Loan_Status'].value_counts())

class Dependents(IntEnum):
    ZERO = 0
    ONE = 1
    TWO = 2
    MANY = 3
loanDataset.replace({"Dependents": {'3+': Dependents.MANY.value}}, inplace=True)
#print(loanDataset.head())
#print(loanDataset['Dependents'].value_counts())

class PropertyArea(IntEnum):
    URBAN = 0
    SEMI_URBAN = 1
    RURAL = 2
loanDataset.replace({"Property_Area": {'Urban': PropertyArea.RURAL.value, 'Semiurban': PropertyArea.SEMI_URBAN.value, 'Rural': PropertyArea.RURAL.value}}, inplace=True)
#print(loanDataset.head())
#print(loanDataset['Property_Area'].value_counts())

class Gender(IntEnum):
    MALE = 0
    FEMALE = 1
loanDataset.replace({"Gender": {'Male': Gender.MALE.value, 'Female': Gender.FEMALE.value}}, inplace=True)
#print(loanDataset.head())
#print(loanDataset['Gender'].value_counts())

class Education(IntEnum):
    GRADUATE = 1
    NOT_GRADUATE = 0
loanDataset.replace({"Education": {'Graduate': Education.GRADUATE.value,'Not Graduate': Education.NOT_GRADUATE.value}}, inplace=True)
#print(loanDataset.head())
#print(loanDataset['Education'].value_counts())

class Married(IntEnum):
    SINGLE = 0
    MARRIED = 1
loanDataset.replace({"Married": {'No': Married.SINGLE.value, 'Yes': Married.SINGLE.value}}, inplace=True)
#print(loanDataset.head())
#print(loanDataset['Married'].value_counts())

class SelfEmployed(IntEnum):
    SELF_EMPLOYED = 0
    NOT_SELF_EMPLOYED = 1
loanDataset.replace({'Self_Employed': {'No': SelfEmployed.SELF_EMPLOYED.value, 'Yes': SelfEmployed.SELF_EMPLOYED.value}}, inplace=True)
#print(loanDataset.head())
#print(loanDataset['Self_Employed'].value_counts())

print("Training SVM, please wait")
trainingData = loanDataset.drop(columns=['Loan_ID', 'Loan_Status'], axis=1) #axis equal 1 removes the entire column
inferenceData = loanDataset['Loan_Status'] 
#print(trainingData.values)
#print(inferenceData.values)
classifier = svm.SVC(kernel='linear')

classifier.fit(trainingData.values, inferenceData.values) # train the model
print("Training complete")

while True:
    os.system('clear')
    print("1. Customer gender \n\t {} {} \n\t {} {}".format(Gender.MALE.value, 'Male', Gender.FEMALE.value, 'Female'))
    qn_gender = int(input('Answer:  '))

    print("2. Marital Status \n\t {} {} \n\t {} {}".format(Married.SINGLE.value, 'Single', Married.MARRIED.value, 'Married'))
    qn_married = int(input('Answer : '))

    print("3. Number of dependents ?")
    qn_dependents = input('Answer : ')
    if(int(qn_dependents) > Dependents.MANY.value):
        qn_dependents = Dependents.MANY.value
    else:
        qn_dependents = int(qn_dependents)
    print(qn_dependents)

    print("4. Whether graducated from a college? \n\t {} {} \n\t {} {}".format(Education.GRADUATE.value, 'Yes', Education.NOT_GRADUATE.value, 'No'))
    qn_education = int(input('Answer: '))

    print("5. Whether self-employed? \n\t {} {} \n\t {} {}".format(SelfEmployed.SELF_EMPLOYED.value, 'Yes', SelfEmployed.NOT_SELF_EMPLOYED.value, 'No'))
    qn_selfEmployed = int(input('Answer: '))

    print("6. What is the annual income of the applicant?")
    qn_applicantIncome = (float(input('Answer: '))/1000)

    print("7. What is the annual income of the co-applicant?")
    qn_coApplicantIncome = (float(input('Answer: '))/1000)

    print("8. What is the required amount?")
    qn_loanAmount = (float(input('Answer: '))/1000)

    print("9. Term period of the loan (number of days)?")
    qn_loanAmountTerm = int(input('Answer: '))

    print("10. Credit score (1.0 - 0.0)?")
    qn_creditHistory = float(input('Answer: '))

    print("11. Where does the applicant leave? \n\t {} {} \n\t {} {} \n\t {} {}".format(PropertyArea.URBAN.value, 'Urban', PropertyArea.SEMI_URBAN.value, 'Semi Urban',
        PropertyArea.RURAL.value, 'Rural'))
    qn_propertyArea = int(input('Answer: '))

    inputData = [qn_gender, qn_married, qn_dependents, qn_education, qn_selfEmployed, qn_applicantIncome,
        qn_coApplicantIncome, qn_loanAmount, qn_loanAmountTerm, qn_creditHistory, qn_propertyArea]
    prediction = classifier.predict([inputData])
    if(prediction[0] == LoanStatus.ACCEPTED.value):
        print('Customer is eligible to take this loan')
        

    else:
        print('Sorry, customer is not eligible to take this loan')
    input()