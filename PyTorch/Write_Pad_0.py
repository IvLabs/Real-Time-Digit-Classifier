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

drawing = False
pt_arr = []
l = 0


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, padding=2),
            torch.nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, padding=2),
            torch.nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.dropout = nn.Dropout()

        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # print(out.shape)
        out = out.reshape((-1, 64 * 7 * 7))
        # print(out.shape)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.fc2(out)

        return out


MODEL_STORE_PATH = "/media/khurshed2504/Data/PycharmProjects/ML_temp0/PyTorch Models/MNIST 0"
model = ConvNet()
model.load_state_dict(torch.load(MODEL_STORE_PATH))
model.eval()

def draw(event, x, y, flags, param):
    global l, pt_arr, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        pt_arr.append((x, y))
        l = l + 1

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            pt_arr.append((x, y))
            cv2.line(pad, pt_arr[l - 1], pt_arr[l], color=0, thickness=1)
            l = l + 1

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False




pad = np.ones((28, 28))*255

cv2.namedWindow('Pad')
ans = np.ones((50,250))*255
cv2.setMouseCallback('Pad', draw)
font = cv2.FONT_HERSHEY_SIMPLEX
while(1):
    cv2.imshow('Pad', pad)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    if k == 27:
        ans = np.ones((50,250))*255
        x = torch.from_numpy(255 - pad)
        x = x.reshape(1, 1, 28, 28)
        x = x.type(torch.float32)
        test_out = model(x)
        _, predicted = torch.max(test_out.data, 1)
        cv2.putText(ans, 'I think its a : ' + str(predicted.item()), (25,25), font, 0.5, 0, thickness=1,lineType=cv2.LINE_AA)
        pad = np.ones((28,28))*250


    cv2.imshow('Pad', pad)
    cv2.imshow('Answer', ans)