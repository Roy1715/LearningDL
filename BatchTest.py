import numpy as np
import pickle
from dataset.mnist import load_mnist
from SigmoidFunc import sigmoid
from SoftMaxFunc import softMax_function

def get_data():
    (x_train,t_train),(x_test,t_test)=\
    load_mnist(flatten=True,normalize=False)

    return x_test,t_test

def init_network():
    with open("ch03/sample_weight.pkl",'rb') as f:
        network= pickle.load(f)

    return network

def predict(network,x):
    w1,w2,w3=network['W1'],network['W2'],network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']

    #第1層
    a1=np.dot(x,w1)+b1
    z1=sigmoid(a1)

    #第2層
    a2=np.dot(z1,w2)+b2
    z2=sigmoid(a2)

    #出力層
    a3=np.dot(z2,w3)+b3
    y=softMax_function(a3)
    
    return y

x,t=get_data()
network=init_network()

batch_size=100

accuracy_cnt=0

for i in range(0,len(x),batch_size):

    y=predict(network,x[i:i+batch_size])
    p=np.argmax(y,axis=1)

    accuracy_cnt+=np.sum(p==t[i:i+batch_size])
print("Accuracy:" + str(float(accuracy_cnt)/len(x)))        