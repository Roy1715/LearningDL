def AND(x1,x2):
    w1,w2,theta=0.5,0.5,0.7
    tmp=x1*w1+x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

for f1 in range(2):
        for f2 in range(2):
            print("in {},{} → out {}".format(f1,f2,AND(f1,f2)))

       
