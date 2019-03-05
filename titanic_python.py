import pandas as pd
import numpy as np

from pandas import Series, DataFrame

titanic_df = pd.read_csv("C://Users//LENOVO IDEAPAD 320//OneDrive//Desktop//Python_Projects//Titanic//train.csv")

print(titanic_df.head())
print(titanic_df.info())

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x = 'Sex', data = titanic_df)

sns.countplot( x = 'Sex', data = titanic_df, hue = 'Pclass')

sns.countplot('Pclass', data = titanic_df, hue = 'Sex')


def male_female_child(passenger):
    age, sex = passenger
    if age < 16:
        return('child')
    else:
        return(sex)
        
titanic_df['person'] = titanic_df[['Age', 'Sex']].apply(male_female_child, axis = 1)

sns.countplot( x = "Pclass", data = titanic_df, hue = "person")

titanic_df['Age'].hist(bins = 70)
print(titanic_df['Age'].mean())

print(titanic_df['person'].value_counts())

fig = sns.FacetGrid(titanic_df, hue = 'Sex', aspect = 4)
fig.map(sns.kdeplot, 'Age', shade = 'True')

oldest = titanic_df['Age'].max()
fig.set(xlim = (0, oldest))
fig.add_legend()


fig = sns.FacetGrid(titanic_df, hue = 'person', aspect = 4)
fig.map(sns.kdeplot, 'Age', shade = 'True')

oldest = titanic_df['Age'].max()
fig.set(xlim = (0, oldest))
fig.add_legend()

fig = sns.FacetGrid(titanic_df, hue = 'Pclass', aspect = 4)
fig.map(sns.kdeplot, 'Age', shade = 'True')

oldest = titanic_df['Age'].max()
fig.set(xlim = (0, oldest))
fig.add_legend()

deck = titanic_df['Cabin'].dropna()#dropping all the null values

levels = []
for level in deck:
    levels.append(level[0])

cabin_df = DataFrame(levels)
cabin_df.columns = ['Cabin']
#sns.countplot( x = 'Cabin', palette = 'winter_d', data = cabin_df)

cabin_df = cabin_df[cabin_df.Cabin != 'T']
#sns.countplot('Cabin', data = cabin_df, palette = 'summer')

#sns.countplot( x = 'Embarked', data = titanic_df, hue = 'Pclass')

titanic_df['Alone'] = titanic_df.SibSp + titanic_df.Parch

titanic_df['Alone'].loc[titanic_df['Alone']>0]  = 'With Family'
titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Alone'

#sns.countplot( x ='Alone', data = titanic_df, palette = 'Blues')

titanic_df['Survivor'] = titanic_df.Survived.map({0: 'no', 1: 'Yes'})

sns.countplot(x = 'Survivor', data = titanic_df, palette = 'Set1')

sns.factorplot('Pclass', 'Survived', data = titanic_df, hue = 'person')

generations = [10, 20, 40, 60, 80]
 
sns.lmplot('Age', 'Survived', hue = 'Pclass', data = titanic_df, palette = 'winter', x_bins = generations)

sns.lmplot('Age', 'Survived', hue = 'Sex', data = titanic_df, palette = 'winter', x_bins = generations)
