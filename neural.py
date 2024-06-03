import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import pickle

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

def init_params(size):
    # initializing all parameters for the formulas/math
    W1=np.random.rand(10,size) - 0.5 # weights for input layer to hidden layer, -0.5 to make the range [-0.5,0.5] rather than [0,1] as this will be helpful with ReLU function
    b1=np.random.rand(10,1) - 0.5 # biases for input layer to hidden layer
    W2=np.random.rand(10,10) - 0.5 # weights for hidden layer to output layer
    b2=np.random.rand(10,1) - 0.5 # biases for hidden layer to output layer
    
    # # using normalization rather than randomization as it gives better results generally
    # W1 = np.random.normal(size=(10, 784)) * np.sqrt(1./(784))
    # b1 = np.random.normal(size=(10, 1)) * np.sqrt(1./10)
    # W2 = np.random.normal(size=(10, 10)) * np.sqrt(1./20)
    # b2 = np.random.normal(size=(10, 1)) * np.sqrt(1./(784))
    return W1,b1,W2,b2

def ReLU(Z):
    # ReLU gives 0 whenever o/p is less than 0 and gives the number itself when the o/p is greater than 0
    return np.maximum(0,Z)

def softmax(Z):
    # # # normalizes the o/p
    # # return np.exp(Z)/np.sum(np.exp(Z))
    # Z -= np.max(Z, axis=0)  # Subtract max value for numerical stability
    # A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    # return A
    exp = np.exp(Z - np.max(Z)) # the "-np.max(Z)" prevents overfitting(i think?)
    return exp / exp.sum(axis=0)

def forward_prop(W1,b1,W2,b2,X):
    Z1=W1.dot(X)+b1 # using the formula
    A1=ReLU(Z1)
    Z2=W2.dot(A1)+b2
    A2=softmax(Z2)
    return Z1,A1,Z2,A2

def one_hot(Y):
    # one-hot encodes the labels, i.e. turns Y into a column matrix
    ''' return an 0 vector with 1 only in the position corresponding to the value in Y'''
    # return np.eye(10)[Y]
    one_hot_Y=np.zeros((Y.size,Y.max()+1))
    one_hot_Y[np.arange(Y.size),Y]=1
    one_hot_Y=one_hot_Y.T # because we want each column to be an example, we will transpose it
    return one_hot_Y

def derivative_ReLU(Z):
    # derivative of ReLU function
    return Z>0

def back_prop(X, Y, A1, A2, W2, Z1, m):
    # # moves backwards in the neural network to find better weights and biases
    # m=Y.size
    # one_hot_Y=one_hot(Y)
    # dZ2=A2-one_hot_Y
    # dW2=1/m * dZ2.dot(A1.T)
    # db2=1/m * np.sum(dZ2,1)
    # dZ1=W2.T.dot(dZ2)*derivative_ReLU(Z1)
    # dW1=1/m * dZ1.dot(X.T)
    # db1=1/m * np.sum(dZ1,1)
    # return dW1,db1,dW2,db2

    one_hot_Y = one_hot(Y)
    dZ2 = 2*(A2 - one_hot_Y) #10,m
    dW2 = 1/m * (dZ2.dot(A1.T)) # 10 , 10
    db2 = 1/m * np.sum(dZ2,1) # 10, 1
    dZ1 = W2.T.dot(dZ2)*derivative_ReLU(Z1) # 10, m
    dW1 = 1/m * (dZ1.dot(X.T)) #10, 784
    db1 = 1/m * np.sum(dZ1,1) # 10, 1

    return dW1, db1, dW2, db2

def update_params(W1,b1,W2,b2,dW1,db1,dW2,db2,alpha):
    # updates the weights and biases, where alpha is a predefined learning rate
    W1 -= alpha * dW1
    b1 -= alpha * np.reshape(db1,(10,1))
    W2 -= alpha * dW2
    b2 -= alpha * np.reshape(db2,(10,1))
    return W1,b1,W2,b2

def get_predictions(A2):
    # returns the predictions of the neural network
    return np.argmax(A2,0)

def get_accuracy(predictions,Y):
    # returns the accuracy of the neural network
    print(predictions,Y)
    return np.sum(predictions==Y)/Y.size

def gradient_descent(X,Y,alpha,iterations):
    size,m=X.shape
    W1,b1,W2,b2=init_params(size) # initializing the parameters
    for i in range(iterations):
        # runs the loop for given iterations, moving forward and backward, updating the params every time
        Z1,A1,Z2,A2=forward_prop(W1,b1,W2,b2,X)
        dW1,db1,dW2,db2=back_prop(X, Y, A1, A2, W2, Z1, m)
        W1,b1,W2,b2=update_params(W1,b1,W2,b2,dW1,db1,dW2,db2,alpha)

        # if i%50==0:
        #     print(f"Iteration: {i} / {iterations}")
        #     print("Accuracy: ",get_accuracy(get_predictions(A2),Y))

        if (i+1) % int(iterations/20) == 0:
            print(f"Iteration: {i+1} / {iterations}")
            prediction = get_predictions(A2)
            print(f'{get_accuracy(prediction, Y):.3%}')

    return W1,b1,W2,b2

# W1,b1,W2,b2=gradient_descent(X_train,Y_train,iterations=1000,alpha=0.1)

### First few attempts were miserable so I added a few more things ###

def make_predictions(X, W1 ,b1, W2, b2):
    _, _, _, A2 = forward_prop(W1,b1,W2,b2,X)
    predictions = get_predictions(A2)
    return predictions

def show_prediction(index,X, Y, W1, b1, W2, b2):
    vect_X = X[:, index,None]
    prediction = make_predictions(vect_X, W1, b1, W2, b2)
    label = Y[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = vect_X.reshape((WIDTH, HEIGHT)) * SCALE_FACTOR

    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

## Main ##

(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
SCALE_FACTOR = 255 # TRES IMPORTANT SINON OVERFLOW SUR EXP
WIDTH = X_train.shape[1]
HEIGHT = X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0],WIDTH*HEIGHT).T / SCALE_FACTOR
X_test = X_test.reshape(X_test.shape[0],WIDTH*HEIGHT).T  / SCALE_FACTOR

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.5, 1000)
with open("trained_params.pkl","wb") as dump_file:
    pickle.dump((W1, b1, W2, b2),dump_file)

with open("trained_params.pkl","rb") as dump_file:
    W1, b1, W2, b2=pickle.load(dump_file)
show_prediction(0,X_test, Y_test, W1, b1, W2, b2)
show_prediction(1,X_test, Y_test, W1, b1, W2, b2)
show_prediction(2,X_test, Y_test, W1, b1, W2, b2)
show_prediction(100,X_test, Y_test, W1, b1, W2, b2)
show_prediction(200,X_test, Y_test, W1, b1, W2, b2)

## This is the final version, seems to be getting about 90% accuracy, which is very good.
## This was my first time making a neural network, with help of course.

