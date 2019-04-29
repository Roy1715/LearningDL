import sys,os
import numpy as np
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image


def image_show(img):
    pil_img=Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train,t_train),(x_test,t_test)=\
    load_mnist(flatten=True,normalize=False)
# print(x_train.shape)
# print(t_train.shape)
# print(x_test.shape)
# print(t_test.shape)

img =x_train[0]
label=t_train[0]

print(img.shape)
print(label)

img=img.reshape(28,28)
image_show(img)