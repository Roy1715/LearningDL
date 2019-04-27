import numpy as np


def softMax_function(a):
    c=np.max(a)
    exp_a=np.exp(a-c)
    #print("\nexp_a:"+str(exp_a))
    sum_exp_a=np.sum(exp_a)
    #print("sum_exp_a:"+str(sum_exp_a))
    Y=exp_a/sum_exp_a
    #print("Y:"+str(Y)+"\n")

    return Y
   

