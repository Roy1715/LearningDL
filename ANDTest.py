import numpy as np

def AND(x1,x2):

    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.7
    tmp=np.sum(x*w)+b

    if tmp > 0:
        return 1
    else :
        return 0

print("AND Start")
for f1 in range(2):
        for f2 in range(2):
            print("in {},{} → out {}".format(f1,f2,AND(f1,f2)))
print("AND End")
       
