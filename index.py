# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv('~/PycharmProjects/titanic/data/train.csv')
test_df = pd.read_csv('~/PycharmProjects/titanic/data/test.csv')
combine = [train_df, test_df]

# Train_df: ['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch'
# 'Ticket' 'Fare' 'Cabin' 'Embarked']

# Create Survival charts

survivalBySex=train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived',ascending=False)
survivalByEmbarkation=train_df[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived',ascending=False)
survivalByClass=train_df[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived',ascending=False)
survivalBySibSp=train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived',ascending=False)
survivalByParch=train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived',ascending=False)

print(survivalBySex)
print(survivalByEmbarkation)
print(survivalByClass)
print(survivalBySibSp)
print(survivalByParch)

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Pclass',bins=3)

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Pclass')