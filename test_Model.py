import torch
from torch import nn
from torchvision import models
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import time
# print(torch.cuda.is_available())  # true 查看GPU是否可用
# print(torch.cuda.device_count())  # GPU数量， 1
# print(torch.cuda.current_device())  # 当前GPU的索引， 0
# print(torch.cuda.get_device_name(0))  # 输出GPU名称
face_label = ['negative', 'positive', 'surprise']
modelpath = 'Model/Hybird_all/model_12.pth'
testpath = 'test_dataset'
model = torch.load(modelpath)
model = model.cuda()
model.eval()
img_name = os.listdir(testpath)
img_path = [os.path.join(testpath, img_name[i]) for i in range(len(img_name))]
# define the transform
transformer = transforms.Compose([transforms.ToPILImage(),
                                  transforms.Resize((224, 224)),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                       std=[0.229, 0.224, 0.225])])
time1 = time.time()

# read img frome test_dataset

img_on0 = cv2.imread(img_path[0])
img_apex0 = cv2.imread(img_path[1])
img_off0 = cv2.imread(img_path[2])
print(img_on0.shape)
print(img_apex0.shape)
print(img_off0.shape)
img_on0 = img_on0[:, :, ::-1]
img_apex0 = img_apex0[:, :, ::-1]
img_off0 = img_off0[:, :, ::-1]

# transform the img
img_on0 = transformer(img_on0)
img_apex0 = transformer(img_apex0)
img_off0 = transformer(img_off0)
img_on1 = img_on0
img_on2 = img_on0
img_on3 = img_on0
img_off1 = img_off0
img_off2 = img_off0
img_off3 = img_off0
ALL = torch.cat((img_on0, img_on1, img_on2, img_on3, img_apex0,
                img_off0, img_off1, img_off2, img_off3), dim=0)

img_on0 = ALL[0:3, :, :]
img_on1 = ALL[3:6, :, :]
img_on2 = ALL[6:9, :, :]
img_on3 = ALL[9:12, :, :]
img_apex0 = ALL[12:15, :, :]
img_off0 = ALL[15:18, :, :]
img_off1 = ALL[18:21, :, :]
img_off2 = ALL[21:24, :, :]
img_off3 = ALL[24:27, :, :]
# let the img in the cuda
# add the batch dimension
img_on0 = img_on0.unsqueeze(0)
img_on1 = img_on1.unsqueeze(0)
img_on2 = img_on2.unsqueeze(0)
img_on3 = img_on3.unsqueeze(0)
img_apex0 = img_apex0.unsqueeze(0)
img_off0 = img_off0.unsqueeze(0)
img_off1 = img_off1.unsqueeze(0)
img_off2 = img_off2.unsqueeze(0)
img_off3 = img_off3.unsqueeze(0)
img_on0 = img_on0.cuda()
img_on1 = img_on1.cuda()
img_on2 = img_on2.cuda()
img_on3 = img_on3.cuda()
img_apex0 = img_apex0.cuda()
img_off0 = img_off0.cuda()
img_off1 = img_off1.cuda()
img_off2 = img_off2.cuda()
img_off3 = img_off3.cuda()
# let face_label in the cuda
# predict the img
print('time to preprocess: ', time.time() - time1, 's')
preprocess_time = time.time() - time1

total_time = 0
for i in range(100):
    time2 = time.time()
    with torch.no_grad():
        output = model(img_on0, img_on1, img_on2, img_on3, img_apex0,
                       img_off0, img_off1, img_off2, img_off3, False)
        print(time.time() - time2, 's')
        total_time += time.time() - time2
        _, pred = torch.max(output, 1)
        #print('time to predict: ', time.time() - time2, 's')
        # change the tensor to numpy
        pred = (pred.cpu().numpy())[0]
        print(face_label[pred])

print('total time: ', time.time() - time1, 's')
print('average time: ', total_time/100, 's')
