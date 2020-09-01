import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def npRelu(X):
    return np.maximum(0,X)

def npLRelu(X):
    return np.maximum(0.01*X,X)


def npSigmoid(X):
    return 1 / (1 + np.exp(-1 * X))


def para_init(n_X,m):
    W1 = np.random.randn(3,n_X)*0.01
    b1 = np.zeros((3,1))

    W2 = np.random.randn(1,3)*0.01
    b2 = 0.0

    return W1, b1, W2, b2


def forward_prop(W1,W2,b1,b2,X):
    Z1 = np.dot(W1, X) + b1
    A1 = npLRelu(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = npSigmoid(Z2)

    cache={"Z1":Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return cache


def cost(A, Y, m):
    c = (-1/m)*(np.dot(Y,np.log(A).T)+np.dot((1-Y),np.log(1-A).T))
    return c


def dRel(X):
    return np.maximum(0,X/np.abs(X))


def update(W1,W2,b1,b2,m,cache,lr):
    Z1 = cache["Z1"]
    Z2 = cache["Z2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = (A2 - Y)
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = np.sum(dZ2)/m

    dZ1 = dRel(Z1)*np.dot(W2.T,dZ2)
    dW1 = (1/m)*np.dot(dZ1,X.T)
    db1 = np.sum(dZ1)/m

    W2 = W2 - lr *dW2
    b2 = b2 - lr * db2
    W1 = W1 - lr*dW1
    b1 = b1 - lr*db1

    return W1, b1, W2, b2

def learn(X,Y,n_X,m,num,lr,show):
    costs = []
    W1, b1, W2, b2 = para_init(n_X,m)

    for i in range(num):
        f_cache = forward_prop(W1,W2,b1,b2,X)
        Z1 = f_cache["Z1"]
        Z2 = f_cache["Z2"]
        A1 = f_cache["A1"]
        A2 = f_cache["A2"]

        predict = np.ceil(A2-0.5)

        costs.append(cost(A2,Y,m))
        W1, b1, W2, b2 = update(W1,W2,b1,b2,m,f_cache,lr)

        if show and (i % 1000==0):
            print("Cost is : ",costs[i-1])
            percent = np.sum(np.abs(predict-Y))
            print("Percentage Accuracy after",i,"iterations is: ",100.00-percent)
    return W1, b1, W2, b2


arr = pd.read_csv("bindata.csv", sep=',', header=None,error_bad_lines=False)
# print(arr)
arr = np.array(arr)

X = arr[:,:-1]
X = X.T
print("X Shape : ",X.shape)
n_X = X.shape[0]
m = X.shape[1]

Y = arr[:,-1].reshape(1,m)
print("Y Shape : ",Y.shape)

W1, b1, W2, b2 = learn(X,Y,n_X,m,10000,0.0097,show=True)

print("W1 : ", W1)
print("b1 : ", b1)
print("W2 : ", W2)
print("b2 : ", b2)
