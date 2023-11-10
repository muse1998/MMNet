import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import os
import CASME3_7
import seaborn as sns
import time
import pandas as pd
torch.set_printoptions(precision=3, edgeitems=14, linewidth=350)
criterion = torch.nn.CrossEntropyLoss()
path = 'Model/casmeiii7'
path_save = 'Model/casmeiii7_all'
if(os.path.exists(path_save) == False):
    os.mkdir(path_save)
modellist = os.listdir(path)
# sort the model list as the order of epoch
modellist.sort(key=lambda x: int(x[11:-4]))
TP_C_total = [0, 0, 0,0,0,0,0]
TP_total = [0, 0, 0,0,0,0,0]
running_loss_t = []
result_file = pd.DataFrame(columns=['model', 'loss', 'acc', 'TP_C', 'TP'])
loso=['spNO.1', 'spNO.2', 'spNO.3', 'spNO.5', 'spNO.6', 'spNO.7', 'spNO.8', 'spNO.9', 'spNO.10', 'spNO.11', 'spNO.12', 'spNO.13', 'spNO.14', 'spNO.15', 'spNO.17', 'spNO.39', 'spNO.40', 'spNO.41', 'spNO.42', 'spNO.77', 'spNO.138', 'spNO.139', 'spNO.142', 'spNO.143', 'spNO.144', 'spNO.145', 'spNO.146', 'spNO.147', 'spNO.148', 'spNO.149', 'spNO.150', 'spNO.152', 'spNO.153', 'spNO.154', 'spNO.155', 'spNO.157', 'spNO.159', 'spNO.160', 'spNO.161', 'spNO.162', 'spNO.163', 'spNO.165', 'spNO.166', 'spNO.167', 'spNO.168', 'spNO.169', 'spNO.170', 'spNO.171', 'spNO.172', 'spNO.173', 'spNO.174', 'spNO.175', 'spNO.176', 'spNO.177', 'spNO.178', 'spNO.179', 'spNO.180', 'spNO.181', 'spNO.182', 'spNO.183', 'spNO.184', 'spNO.185', 'spNO.186', 'spNO.187', 'spNO.188', 'spNO.189', 'spNO.190', 'spNO.192', 'spNO.194', 'spNO.195', 'spNO.196', 'spNO.197', 'spNO.198', 'spNO.200', 'spNO.201', 'spNO.202', 'spNO.203', 'spNO.204', 'spNO.206', 'spNO.207', 'spNO.208', 'spNO.209', 'spNO.210', 'spNO.211', 'spNO.212', 'spNO.213', 'spNO.214', 'spNO.216', 'spNO.217']
for i in range(len(modellist)):
    test = CASME3_7.RafDataSet(
        '/home/drink36/Desktop/apex_frame_take/casme3_result_2', phase='test', num_loso=loso[i],
        transform=transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])]))
    test_loader = torch.utils.data.DataLoader(test,
                                              batch_size=1,  # 32
                                              #  collate_fn=lambda x: default_collate(
                                              #      x).to(device),
                                              shuffle=False,
                                              pin_memory=True,
                                              )
    model = CASME3_7.MMNet()
    # model = torch.load(os.path.join(path_save, modellist[i]))
    model.load_state_dict(torch.load(os.path.join(path, modellist[i])))
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        iter_cnt = 0
        bingo_cnt = 0
        sample_cnt = 0
        pos_label = []
        pos_pred = []
        TP = [0, 0, 0,0,0,0,0]
        TP_C = [0, 0, 0,0,0,0,0]
        for batch_i, (
            image_on0, image_on1, image_on2, image_on3, image_apex0, image_off0, image_off1, image_off2,
            image_off3, label_all,label_au
        ) in enumerate(test_loader):
            batch_sz = image_on0.size(0)
            b, c, h, w = image_on0.shape
            image_on0 = image_on0.cuda()
            image_on1 = image_on1.cuda()
            image_on2 = image_on2.cuda()
            image_on3 = image_on3.cuda()
            image_apex0 = image_apex0.cuda()
            image_off0 = image_off0.cuda()
            image_off1 = image_off1.cuda()
            image_off2 = image_off2.cuda()
            image_off3 = image_off3.cuda()
            label_all = label_all.cuda()
            label_au = label_au.cuda()
            time1 = time.time()
            ALL = model(image_on0, image_on1, image_on2, image_on3, image_apex0,
                        image_off0, image_off1, image_off2, image_off3, False)
            loss = criterion(ALL, label_all)
            running_loss += loss.item()
            iter_cnt += 1
            _, pred = torch.max(ALL, 1)
            corret_num = torch.eq(pred, label_all)
            bingo_cnt += corret_num.sum().cpu()
            sample_cnt += batch_sz
            # 0:negative, 1:positive, 2:surprise
            for cls in range(7):
                for element in pred:
                    if element == cls:
                        pos_pred.append(cls)
                for element in label_all:
                    if element == cls:
                        pos_label.append(cls)
                        TP_C[cls] = TP_C[cls] + 1
                for elementp, elementl in zip(pred, label_all):
                    if elementp == elementl and elementp == cls:
                        TP[cls] = TP[cls] + 1
        running_loss = running_loss / iter_cnt
        acc = bingo_cnt.float() / float(sample_cnt)
        acc = acc.cpu()
        acc = np.around(acc.numpy(), 4)
        # save the model
        torch.save(model, os.path.join(path_save, modellist[i]))
        # save the result to dataframe
        result_file.loc[i] = [modellist[i], running_loss, acc, TP_C, TP]
        print('model:', modellist[i])
        print('Test Loss: {:.4f}, Test Acc: {:.4f}'.format(running_loss, acc))
        running_loss_t.append(running_loss)
        print('TP_C:', TP_C)
        print('TP:', TP)
        print('------------------------------------')
        TP_C_total = [x + y for x, y in zip(TP_C_total, TP_C)]
        TP_total = [x + y for x, y in zip(TP_total, TP)]
print('TP_C_total:', TP_C_total)
print('TP_total:', TP_total)
print('TP/TP_C:', [float(x) / float(y)
                   for x, y in zip(TP_total, TP_C_total)])
print(sum(TP_total) / sum(TP_C_total))
# print('running_loss_t:', running_loss_t)
# sort the model list as the order of loss and keep the top 5
modellist = [x for _, x in sorted(zip(running_loss_t, modellist))]
# average the running_loss
running_loss_t = [sum(running_loss_t) / len(running_loss_t)]
# add new row to dataframe
result_file.loc[len(modellist)] = ["ALL",running_loss_t ,sum(TP_total) / sum(TP_C_total), TP_C_total, TP_total]
# save the result to csv
result_file.to_csv('result_file.csv', index=False)
