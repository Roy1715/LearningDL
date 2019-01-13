import numpy as np
from ORTest import OR
from ANDTest import AND
from NANDTest import NAND

def XOR(x1, x2):
     
    tmp=AND(OR(x1,x2),NAND(x1,x2))
    return tmp
    
    
for f1 in range(2):
    for f2 in range(2):
        print("XOR in {} {} → out {}".format(f1,f2,XOR(f1,f2)))
