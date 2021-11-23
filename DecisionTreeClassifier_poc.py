# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 11:35:23 2020
This file uses decision tree classifier to classify the outcome of the diabetes
@author: Ashok Chinnaswamy 

"""

# Import the dependencies / libraries
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

#Create a dataframe from the Diabetes csv file
df = pd.read_csv('C:/AshokC/Documents/Diabetestrain.csv')

#print the first 5 rows of the data set
print(df.head())

# Split your data into X_train and Y_train
X_train = df.loc[:,'Pregnancies':'Age'] #Gets all the rows in the dataset from column 'Pregnancies' to column 'Age'
Y_train = df.loc[:,'Outcome'] #Gets all of the rows in the dataset from column 'Outcome'

# The actual decision tree classifier
tree = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)     

# Train the model
tree.fit(X_train, Y_train)

# Make your Input prediction array
prediction = tree.predict([[8,154,78,32,0,32.4,0.443,45],[6,190,92,0,0,35.5,0.278,66],[10,101,76,48,180,32.9,0.171,63]])

#Print the prediction
#print('Printing the prediction: ')
print(prediction)

