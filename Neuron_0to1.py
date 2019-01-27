import numpy as np
from SigmoidFunc import sigmoid


X=np.array([1.0,0.5])
print("X"+str(X))
W1=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1=np.array([0.1,0.2,0.3])
print(W1.shape)
print(X.shape)
print(B1.shape)

A1=np.dot(X,W1)+B1
print("A1:"+str(A1))

Z1=sigmoid(A1)
print("↓H(X)\nZ1:"+str(Z1))

W2=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2=np.array([0.1,0.2])
A2=np.dot(Z1,W2)+B2
print("A2:"+str(A2))
Z2=sigmoid(A2)

def identity_function(x):
    return x

W3=np.array([[0.1,0.3],[0.2,0.4]])
B3=np.array([0.1,0.2])

A3=np.dot(Z2,W3)+B3
print("A3:"+str(A3))

Y=identity_function(A3)
print("Y:"+str(Y))