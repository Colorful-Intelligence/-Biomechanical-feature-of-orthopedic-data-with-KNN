# -*- coding: utf-8 -*-
"""
Created on Sun May 30 13:32:04 2021

@author: bymeh
"""

#%% Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

import warnings
warnings.filterwarnings('ignore')

#%%

# read the dataset

data =pd.read_csv("dataset.csv")

#%% Exploratory Data Analysis
data.head() # first 5 rows
data.info()
data.describe
data.columns

#%% 

# in this dataset , target column is class , I'm going to change column name as a "target"

data = data.rename({"class":"Target"},axis = 1)

Abnormal = data[data.Target == "Abnormal"]
Normal = data[data.Target == "Normal"]

#%% Visualize
plt.scatter(Abnormal.lumbar_lordosis_angle,Abnormal.sacral_slope,color = "red",alpha = 0.3)
plt.scatter(Normal.lumbar_lordosis_angle,Normal.sacral_slope,color = "green",alpha = 0.3)
plt.xlabel("lumbar_lordosis_angle")
plt.ylabel("sacral_slope")
plt.legend()
plt.show()


#%% 
data.Target = [1 if each == "Normal" else 0 for each in data.Target]
y = data.Target.values
x_data = data.drop(["Target"],axis = 1) # axis = 1 -> column , axis = 0 -> row


#%% Normalization 

x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

#%% train-test split

# train ile modelimizi test edeceğiz , test ile predict yapacağız
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =  train_test_split(x,y,test_size = 0.3,random_state = 1)



#%% KNN Model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=21) # n_neighbors = k value
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)

print("{} nn score {}".format(21,knn.score(x_test,y_test)))



#%% find k value

score_list = []

for each in range(1,150):
    knn2 = KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))


plt.plot(range(1,150),score_list)
plt.title("K-Value & Accuracy")
plt.xlabel("K-Value")
plt.ylabel("Accuracy")
plt.show()







