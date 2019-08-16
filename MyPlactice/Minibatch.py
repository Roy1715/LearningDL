import numpy as np
from dataset.mnist import load_mnist
from ch04.two_layer_net import TwoLayerNet
import matplotlib.pyplot as plt

(x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,one_hot_label=True)


#ハイパーパラメータ　
iters_num=10000
train_size=x_train.shape[0]
batch_size=100
learning_rate=0.1

train_loss_list =[]
train_acc_list=[]
test_acc_list=[]
time_iter_epoch=[]

#1エポックあたりの繰り返し数
iter_per_epoch=max(train_size/batch_size,1)

network=TwoLayerNet(input_size=784,hidden_size=50,output_size=10)

for i in range(iters_num):
    #ミニバッチの実装
    batch_mask=np.random.choice(train_size,batch_size)
    x_batch=x_train[batch_mask]
    t_batch=t_train[batch_mask]

    #勾配の計算
    grad=network.numerical_gradient(x_batch,t_batch)
    #grad=network.gradient(x_batch,t_batch) #高速版!

    #パラメータの更新
    for key in ('W1','b1','W2','b2'):
        network.params[key] -=learning_rate*grad[key]

    #学習経過の記録
    loss=network.loss(x_batch,t_batch)
    train_loss_list.append(loss)

    #1エポックごとの認識精度を計算
    if i % iter_per_epoch==0:
        train_acc=network.accuracy(x_train,t_train)
        test_acc=network.accuracy(x_test,t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        time_iter_epoch.append(i)
        print("train acc, test acc | "+str(train_acc)+","+str(test_acc))

plt.plot(time_iter_epoch,train_acc_list,label="train acc")
plt.plot(time_iter_epoch,test_acc_list,linestyle="--",label="test acc")
plt.xlabel("Learning Time")
plt.ylabel("Accuracy")
plt.title('Learning Status')
plt.legend()
plt.show()