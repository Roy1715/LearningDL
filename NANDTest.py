import numpy as np

def NAND(x1, x2):
    x=np.array([x1,x2])
    w=np.array([-0.5,-0.5])
    b=0.7
    tmp = np.sum(x*w)+b

    if tmp<=0:
        return 0
    else :
        return 1

for f1 in range(2):
    for f2 in range(2):
        print("in {},{} → out {}".format(f1,f2,NAND(f1,f2)) )

