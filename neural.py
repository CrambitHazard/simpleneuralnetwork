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

data_train=data[1000:m].T  # creating our transposed training dataset
Y_train=data_train[0]
X_train=data_train[1:n]


def init_params():
    # initializing all parameters for the formulas/math
    W1=np.random.randn(10,784) - 0.5 # weights for input layer to hidden layer, -0.5 to make the range [-0.5,0.5] rather than [0,1] as this will be helpful with ReLU function
    b1=np.random.randn(10,1) - 0.5 # biases for input layer to hidden layer
    W2=np.random.randn(10,10) - 0.5 # weights for hidden layer to output layer
    b2=np.random.randn(10,1) - 0.5 # biases for hidden layer to output layer
    return W1,b1,W2,b2