from PIL import Image
import numpy as np

w1 = []
w2 = []

b1 = [0,2,0,0,-1,-2,-2,-1]
b2 = [0,0.2]

iN = []
oN = 0

def SetWeightsData(fileName):
    w=[]
    im = Image.open(fileName, 'r')
    piexels = list(im.getdata())

    for pixel in piexels:
        if   (pixel == (255,  0, 0)):
            w.append(-1)
        elif (pixel == (0,  255, 0)):
            w.append(1)
        else:
            w.append(0)
    return w

def SetInputData(fileName):
    im = Image.open(fileName, 'r')
    piexels = list(im.getdata())

    for pixel in piexels:
        if (pixel == (255, 255, 255)):
            iN.append(1)
        else:
            iN.append(0)

def FeedForward(weights, activations, biases):
    return 1/(1 + np.exp(-(np.dot(weights, activations) + biases)))

w1.extend(SetWeightsData("DataSet\Sign\Layer2\WeightsM1.bmp"))
w1.extend(SetWeightsData("DataSet\Sign\Layer2\WeightsM2.bmp"))
w1.extend(SetWeightsData("DataSet\Sign\Layer2\WeightsM3.bmp"))
w1.extend(SetWeightsData("DataSet\Sign\Layer2\WeightsM4.bmp"))
w1.extend(SetWeightsData("DataSet\Sign\Layer2\WeightsS1.bmp"))
w1.extend(SetWeightsData("DataSet\Sign\Layer2\WeightsS2.bmp"))
w1.extend(SetWeightsData("DataSet\Sign\Layer2\WeightsS3.bmp"))
w1.extend(SetWeightsData("DataSet\Sign\Layer2\WeightsS4.bmp"))

w2.extend(SetWeightsData("DataSet\Sign\Output\WeightsMinus.bmp"))
w2.extend(SetWeightsData("DataSet\Sign\Output\WeightsMultiply.bmp"))

SetInputData("DataSet\Sign\Input2.bmp")

iN = np.array(iN)
w1 = np.array(w1)
w2 = np.array(w2)
w1 = w1.reshape(8, 100)
w2 = w2.reshape(2, 8)

l2N = FeedForward(w1, iN,  b1)
oN  = FeedForward(w2, l2N, b2)

print(l2N)
print(oN)