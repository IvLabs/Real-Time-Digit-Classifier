import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def npRelu(X):
    return np.maximum(0,X)


def npSigmoid(X):
    return 1 / (1 + np.exp(-1 * X))


def parainit(n_X, m):
    W = np.random.randn(1, n_X)*0.01
    b = 0.0
    return W, b


def forward_prop(W, X,b, activation):
    Z = np.dot(W, X) + b
    if activation == "Relu":
        A = npRelu(Z)
    if activation == "Sigmoid":
        A = npSigmoid(Z)
    if activation == "TanH":
        A = np.tanh(X)
    cache = {"Z": Z}
    pred = np.ceil(A-0.5)
    return A, cache,pred


def cost(A, Y, activation, m):
    if activation == "Sigmoid":
        c = (-1/m)*(np.dot(Y,np.log(A).T)+np.dot((1-Y),np.log(1-A).T))
    if activation == "Relu":
        c = (1 / 2 * m) * np.sum(np.power((A - Y), 2))
    return c

def update(W, A, b, Y, m, activation, lr):

    if activation == "Sigmoid":
        der = (A - Y)
    if activation == "Relu":
        der = A / A
    W = W - lr * (1 / m) * np.dot(der, X.T)
    b = b - np.sum(lr * der)
    return W, b


def learn(W, b, X, m, Y, num, lr, activation, show):
    costs = []
    for i in range(1, num):
        A, f_cache, predict = forward_prop(W, X,b, activation)
        costs.append(cost(A, Y, activation,m))
        W, b = update(W, A, b, Y, m, activation, lr)
        if show and (i % 1000==0):
            print("Cost is : ",costs[i-1])
            percent = np.sum(np.abs(predict-Y))
            print("Percentage Accuracy after",i,"iterations is: ",100.00-percent)
    return W, b


# filename = 'nba.csv'
# raw_data = open(filename)

arr = pd.read_csv("bindata.csv", sep=',', header=None,error_bad_lines=False)
print(arr)
arr = np.array(arr)
# arr = float(np.loadtxt("Temp1.txt",delimiter=','))
X = arr[0:99,0:2]
X = X.T
max=np.max(X)
print(X.shape)
# scaler = MinMaxScaler(feature_range=(0, 1))
# X = scaler.fit_transform(X)



# print('X Shape : ', X.shape)

Y = arr[0:99,2]
Y = Y.T
print('Y Shape : ', Y)

n_X = X.shape[0]
m = X.shape[1]
W, b = parainit(n_X, m)

W, b = learn(W, b, X, m, Y, 10000, 0.012, activation="Sigmoid", show=True)

plt.xlabel("x1")
plt.ylabel("x2")
for i in range (0,99):
    if Y[i] :
        plt.plot(X[0,i], X[1,i], 'go')
    else :
        plt.plot(X[0,i],X[1,i],'r*  ')

print("W is ",W)
print("b is ",b)

w1 = W[0,0]

w2 = W[0,1]

xval = np.array(np.linspace(np.min(X[1,:]),np.max(X[1,:]),1000))
yval =-1* (b +W[0,0]*xval)/W[0,1]
plt.plot(xval, yval, 'b.')
plt.show()
