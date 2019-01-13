import numpy as np

def OR(x1, x2):
    x=np.array([x1,x2])
    w=np.array([0.1,0.1])
    b=-0.09
    tmp=np.sum(x*w)+b

    if tmp>0:
        return 1
    
    else :
        return 0

print("OR Start")
for f1 in range(2):
    for f2 in range(2):
        print("in {} {} → out {}".format(f1,f2,OR(f1,f2)))
print("OR End")