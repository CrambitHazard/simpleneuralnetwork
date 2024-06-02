import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import keras

data=pd.read_csv("C:/Python/datasets/train.csv/train.csv") # this is our training data
# print(data.head()) # used to check the first few rows of the data
data=np.array(data) # data is easier to work in an array

m,n=data.shape # recording the amount of rows and features
np.random.shuffle(data) # randomizing as a pre-step for cross-validation

data_dev=data[0:1000].T # creating our cross validation data to prevent overfitting, also the .T is for easier access in future steps i.e. each column is an example rather than each row
# print(data_dev)

Y_dev=data[0] # first row contains all the examples, i.e. numbers
X_dev=data[1:n] # rest of the rows contain the features, i.e. the values of the pixels