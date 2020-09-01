import cv2
import time


import numpy as np
import torch
import torch.nn as nn

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

MODEL_STORE_PATH = "/media/khurshed2504/Data/Summer Project 2019/PyTorch/PyTorch Models/MNIST 0"


modeltest = ConvNet()
modeltest.load_state_dict(torch.load(MODEL_STORE_PATH))
modeltest.eval()

font = cv2.FONT_HERSHEY_SIMPLEX


def nothing():
    pass


cap = cv2.VideoCapture(0)

cv2.namedWindow('frame')
# cv2.createTrackbar('H', 'frame', 0, 180, nothing)
# cv2.createTrackbar('S', 'frame', 0, 255, nothing)
# cv2.createTrackbar('V', 'frame', 0, 255, nothing)

pad = np.zeros((128, 128), dtype='uint8')
pad_b = np.zeros((480,640,3), dtype='uint8')
pt_arr = []
pt_arr_b = []
l = 0
ans = np.ones((50, 250,3)) * 255


while (1):
    try:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Setting Thresholds
        # h = cv2.getTrackbarPos('H', 'frame')
        # s = cv2.getTrackbarPos('S', 'frame')
        # v = cv2.getTrackbarPos('V', 'frame')

        lower = np.array([110, 50, 50])
        upper = np.array([130, 255, 255])

        thresh = cv2.inRange(hsv, lower, upper)

        thresh = cv2.GaussianBlur(thresh, (3, 3), 0)
        cont, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in cont]

        max_index = np.argmax(areas)
        cnt = cont[max_index]
        cv2.drawContours(frame, [cnt], -1, (255, 0, 0), 2)
        M = cv2.moments(thresh)

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        X = int(cX * 128 / 640)
        Y = int(cY * 128 / 480)

        pt_arr.append((X, Y))
        pt_arr_b.append((cX, cY))

        if l > 0:
            cv2.line(pad, pt_arr[l - 1], pt_arr[l], 255, 7)
            cv2.line(pad_b, pt_arr_b[l - 1], pt_arr_b[l], (255, 255, 255), 7)
            frame = cv2.bitwise_or(frame, pad_b)
        l = l + 1
        k = cv2.waitKey(1) & 0XFF
        if k == 27:
            ans = np.ones((50, 250)) * 255
            sample = cv2.resize(pad, (28, 28))
            x = torch.from_numpy(sample)
            x = x.reshape(1, 1, 28, 28)
            x = x.type(torch.float32)
            outputs_test = modeltest(x)
            _, predicted = torch.max(outputs_test.data, 1)
            cv2.putText(ans, 'I think its a : ' + str(predicted.item()), (25, 25), font, 0.5, (0, 0, 0), thickness=1,
                        lineType=cv2.LINE_AA)
            cv2.imshow('frame', frame)
            time.sleep(1)
            pt_arr = []
            pt_arr_b = []
            l = 0
            pad_b = np.zeros((480, 640, 3), dtype='uint8')
            pad = np.zeros((128, 128), dtype='uint8')

        # cv2.imshow('pad', pad)
        cv2.imshow('frame', frame)
        cv2.imshow('Answer', ans)

        if k == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

        if k == ord('c'):
            cv2.putText(ans, 'Clearing Pad', (25, 25), font, 0.5, 0, thickness=1, lineType=cv2.LINE_AA)
            cv2.imshow('Answer', ans)
            time.sleep(1)
            pt_arr = []
            pt_arr_b = []
            l = 0
            pad = np.zeros((128, 128), dtype='uint8')
            pad_b = np.zeros((480, 640, 3), dtype='uint8')
            ans = np.ones((50, 250, 3)) * 255

    except:
        nothing()
