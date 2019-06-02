import numpy as np
import matplotlib.pyplot as mlt
import pandas as pd
dataset=pd.read_csv("Churn_Modelling.csv")
x=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_2 = LabelEncoder()
x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])
labelencoder_x_1 = LabelEncoder()
x[:, 1] = labelencoder_x_2.fit_transform(x[:, 1])

onehotencoder = OneHotEncoder(categorical_features =[1])
x = onehotencoder.fit_transform(x).toarray()
x=x[:,1:]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train) 
x_test=sc.fit_transform(x_test)


import tensorflow as tf
from tensorflow.keras import layers
