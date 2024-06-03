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


####### Functions #######

def init_params():
    # initializing all parameters for the formulas/math
    # W1=np.random.randn(10,784) - 0.5 # weights for input layer to hidden layer, -0.5 to make the range [-0.5,0.5] rather than [0,1] as this will be helpful with ReLU function
    # b1=np.random.randn(10,1) - 0.5 # biases for input layer to hidden layer
    # W2=np.random.randn(10,10) - 0.5 # weights for hidden layer to output layer
    # b2=np.random.randn(10,1) - 0.5 # biases for hidden layer to output layer
    
    # using normalization rather than randomization as it gives better results generally
    W1 = np.random.normal(size=(10, 784)) * np.sqrt(1./(784))
    b1 = np.random.normal(size=(10, 1)) * np.sqrt(1./10)
    W2 = np.random.normal(size=(10, 10)) * np.sqrt(1./20)
    b2 = np.random.normal(size=(10, 1)) * np.sqrt(1./(784))
    return W1,b1,W2,b2

def ReLU(Z):
    # ReLU gives 0 whenever o/p is less than 0 and gives the number itself when the o/p is greater than 0
    return np.maximum(0,Z)

def softmax(Z):
    # # normalizes the o/p
    # return np.exp(Z)/np.sum(np.exp(Z))
    Z -= np.max(Z, axis=0)  # Subtract max value for numerical stability
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A

def forward_prop(W1,b1,W2,b2,X):
    Z1=W1.dot(X)+b1 # using the formula
    A1=ReLU(Z1)
    Z2=W2.dot(A1)+b2
    A2=softmax(Z2)
    return Z1,A1,Z2,A2

def one_hot(Y):
    # one-hot encodes the labels, i.e. turns Y into a column matrix
    # return np.eye(10)[Y]
    one_hot_Y=np.zeros((Y.size,Y.max()+1))
    one_hot_Y[np.arange(Y.size),Y]=1

# def back_prop(Z1,A1,Z2,A2,W2,Y):


