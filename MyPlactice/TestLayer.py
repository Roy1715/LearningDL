from Layer.MulLayer import MullLayer
from Layer.AddLayer import AddLayer


# Buy Apple Pattern
apple = 100
applefigure = 2

appleLayer = MullLayer()
taxLayer = MullLayer()
tax = 1.1

# forward
print("\n---forward---")
applePrice = appleLayer.forward(apple, applefigure)
print("applePrice:" + str(applePrice))
allprice = int(taxLayer.forward(float(applePrice), tax))

print("allPrice:" + str(allprice))

# backward
print("\n---backward---")
dApplePrice, dtax = taxLayer.backward(1)
print("dtax:"+str(dtax))
dApple, dAppleFigure = appleLayer.backward(dApplePrice)
print("dApple:"+str(dApple))
print("dAppleFigure:"+str(int(dAppleFigure))+"\n")


# Buy Apple&Orenge Pattern

# Number Initialize
appleNum = 2
applePrice = 100

orengeNum = 3
orengePrice = 150

tax = 1.1

# Layer Initialize
LayerAppMul = MullLayer()
LayerOreMul = MullLayer()
LayerBothAdd = AddLayer()
LayerTaxMul = MullLayer()

# Forward
AllAppPrice = LayerAppMul.forward(appleNum, applePrice)
AllOrePrice = LayerOreMul.forward(orengeNum, orengePrice)
BothPrice = LayerBothAdd.forward(AllAppPrice, AllOrePrice)
TaxBothPrice = LayerTaxMul.forward(BothPrice, tax)

print("\n--------------forward------------------")
print("TaxPrice:" + str(int(TaxBothPrice)))

# Backward
print("\n--------------backward------------------")

dout = 1
dBothPrice, dTax = LayerTaxMul.backward(dout)
print("dTax:" + str(dTax))
print("dBothPrice:"+str(dBothPrice))

dAllAppPrice, dAllOrePrice = LayerBothAdd.backward(dBothPrice)
print("{}{}\n{}{}".format("dAllAppPrice:",
                          dAllAppPrice, "dAllOrePrice:", dAllOrePrice))

dappleNum, dapplePrice = LayerAppMul.backward(dAllAppPrice)
print("{}{}\n{}{}".format("dappleNum:", int(
    dappleNum), "dapplePrice:", dapplePrice))

dorengeNum, dorengePrice = LayerOreMul.backward(dAllOrePrice)
print("{}{}\n{}{}\n".format("dorengeNum:",
                            dorengeNum, "dorengePrice:", round(dorengePrice, 1)))
