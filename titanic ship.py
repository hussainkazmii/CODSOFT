import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# data collection and processing
titanic_data = pd.read_csv('/Users/LENOVO/Downloads/Titanic-Dataset.csv')
print(titanic_data.head())

# geting info about data
print(titanic_data.info())
# checking the number of null value in Dataset
# print(titanic_data.isnull().sum())

# handling the missing data, as most of the data is missing in cabin set so we drop it
titanic_data = titanic_data.drop(columns='Cabin', axis=1) #axis=1 for col, axis=0 for row

# for the missing age value we can replace it with the mean of the ages
titanic_data['Age'].fillna(titanic_data['Age'].mean, inplace=True) #used inplace so that it could be saved in my original dataset

# finding the most repeated val in Embarked set
print(titanic_data['Embarked'].mode()) #the mode is S

# replacing the missing val in embarked with mode value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)
print(titanic_data.isnull().sum())
# handled the missing data now data analysis
print(titanic_data.describe()) #gives mean,std,min,max (works for numeric data)

# finding the number of people survived and their sex
print(titanic_data['Survived'].value_counts())
print(titanic_data['Sex'].value_counts())

# for data visualization we use sns library
sns.set()

# making count plot for data visualization
sns.countplot(titanic_data["Survived"])
sns.countplot(titanic_data["Sex"])

# number of survivors based on gender
sns.countplot('Sex', hue='Survived', data=titanic_data)

