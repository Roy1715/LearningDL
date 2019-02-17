import sys,os
import numpy as np

sys.path.append(os.pardir)

from dataset.mnist import load_mnist

(x_train,t_train),(x_test,t_test)=\
    load_mnist(normalize=True,one_hot_label=True)

print(x_train.shape)
print(t_train.shape)    

train_size=t_train.shape[0]
print("t_train.shape[0]:"+str(train_size))

batch_size=10

batch_mask=np.random.choice(train_size,batch_size)
x_batch=x_train[batch_mask]
t_batch=t_train[batch_mask]
print("batch_mask:"+str(batch_mask))
print("x_train[batch_mask]:"+str(x_batch))
print("t_train[batch_mask]:"+str(t_batch))