import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as dt
import pickle as cPickle
import gzip
import cv2
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1,32, 5, 1, padding = 2),
            torch.nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, padding=2),
            torch.nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.dropout = nn.Dropout()

        self.fc1 = nn.Linear(7*7*64, 1000)
        self.fc2 = nn.Linear(1000, 10)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # print(out.shape)
        out = out.reshape((-1,64*7*7))
        # print(out.shape)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.fc2(out)

        return out

MODEL_STORE_PATH = "/media/khurshed2504/Data/PycharmProjects/ML_temp0/PyTorch Models/MNIST 0"

img = cv2.imread('5(0).jpg',1)
img = cv2.resize(img,(28,28))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(3,3),0)
ret,thresh = cv2.threshold(gray,240, 255, cv2.THRESH_BINARY)

cv2.imshow('img', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

x = thresh
x = np.float32(x/255)

modeltest = ConvNet()
modeltest.load_state_dict(torch.load(MODEL_STORE_PATH))
modeltest.eval()


x = torch.from_numpy(x)
x = x.reshape(1,1,28,28)
outputs_test = modeltest(x)
_, predicted = torch.max(outputs_test.data, 1)

print("I Think it's a : ", predicted.item())