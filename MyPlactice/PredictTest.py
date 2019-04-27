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

accuracy_cnt=0

for i in range(len(x)):

    y=predict(network,x[i])
    p=np.argmax(y)

    if p==t[i]:
        accuracy_cnt+=1

print("Accuracy:" + str(float(accuracy_cnt)/len(x)))        
