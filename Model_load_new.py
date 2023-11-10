import torch
import torchvision.transforms as transforms
import numpy as np
import os
import CASME3_7
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import warnings

# Suppress the warning message related to the antialias parameter
warnings.filterwarnings("ignore", message="The default value of the antialias parameter of all the resizing transforms")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(precision=3, edgeitems=14, linewidth=350)
criterion = torch.nn.CrossEntropyLoss()
path = 'Model/casmeiii_5'
path_save = 'Model/casmeiii_5_all'
if(os.path.exists(path_save) == False):
    os.mkdir(path_save)
modellist = os.listdir(path)
# sort the model list as the order of epoch
modellist.sort(key=lambda x: int(x[11:-4]))
true_label = []
pred_label = []
TP_C_total = [0, 0, 0,0,0,0,0]
TP_total = [0, 0, 0,0,0,0,0]
running_loss_t = []
result_file = pd.DataFrame(columns=['model', 'loss', 'acc', 'TP_C', 'TP'])
loso=['spNO.1', 'spNO.2', 'spNO.3', 'spNO.5', 'spNO.6', 'spNO.7', 'spNO.8', 'spNO.9', 'spNO.10', 'spNO.11', 'spNO.12', 'spNO.13', 'spNO.14', 'spNO.15', 'spNO.17', 'spNO.39', 'spNO.40', 'spNO.41', 'spNO.42', 'spNO.77', 'spNO.138', 'spNO.139', 'spNO.142', 'spNO.143', 'spNO.144', 'spNO.145', 'spNO.146', 'spNO.147', 'spNO.148', 'spNO.149', 'spNO.150', 'spNO.152', 'spNO.153', 'spNO.154', 'spNO.155', 'spNO.157', 'spNO.159', 'spNO.160', 'spNO.161', 'spNO.162', 'spNO.163', 'spNO.165', 'spNO.166', 'spNO.167', 'spNO.168', 'spNO.169', 'spNO.170', 'spNO.171', 'spNO.172', 'spNO.173', 'spNO.174', 'spNO.175', 'spNO.176', 'spNO.177', 'spNO.178', 'spNO.179', 'spNO.180', 'spNO.181', 'spNO.182', 'spNO.183', 'spNO.184', 'spNO.185', 'spNO.186', 'spNO.187', 'spNO.188', 'spNO.189', 'spNO.190', 'spNO.192', 'spNO.194', 'spNO.195', 'spNO.196', 'spNO.197', 'spNO.198', 'spNO.200', 'spNO.201', 'spNO.202', 'spNO.203', 'spNO.204', 'spNO.206', 'spNO.207', 'spNO.208', 'spNO.209', 'spNO.210', 'spNO.211', 'spNO.212', 'spNO.213', 'spNO.214', 'spNO.216', 'spNO.217']
classes=['happy','fear','disgust','surprise','Others']
for i in range(len(modellist)):
    test = CASME3_7.RafDataSet(
        '/home/drink36/Desktop/apex_frame_take/casme3_result_2', phase='test', num_loso=loso[i],
        transform=transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize((224, 224),antialias=True),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])]),)
        
    test_loader = torch.utils.data.DataLoader(test,
                                              batch_size=1,  # 32
                                              #  collate_fn=lambda x: default_collate(
                                              #      x).to(device),
                                              shuffle=False,
                                              pin_memory=True,
                                              )
    if(test_loader.__len__() == 0):
        continue
    model = CASME3_7.MMNet()
    # model = torch.load(os.path.join(path_save, modellist[i]))
    model.load_state_dict(torch.load(os.path.join(path, modellist[i])))
    model = model.to(device)
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
            image_on0 = image_on0.to(device)
            image_on1 = image_on1.to(device)
            image_on2 = image_on2.to(device)
            image_on3 = image_on3.to(device)
            image_apex0 = image_apex0.to(device)
            image_off0 = image_off0.to(device)
            image_off1 = image_off1.to(device)
            image_off2 = image_off2.to(device)
            image_off3 = image_off3.to(device)
            label_all = label_all.to(device)
            label_au = label_au.to(device)
            time1 = time.time()
            ALL = model(image_on0, image_on1, image_on2, image_on3, image_apex0,
                        image_off0, image_off1, image_off2, image_off3, False)
            loss = criterion(ALL, label_all)
            running_loss += loss.item()
            iter_cnt += 1
            _, pred = torch.max(ALL, 1)
            true_label+=label_all.cpu().numpy().tolist()
            pred_label+=pred.cpu().numpy().tolist()
            batch_correct = torch.eq(pred, label_all).sum().float().item()
            bingo_cnt += batch_correct
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
        acc = bingo_cnt / float(sample_cnt)
        acc = np.around(acc, 4)
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
# print('TP/TP_C:', [float(x) / float(y)
#                    for x, y in zip(TP_total, TP_C_total)])
# UAR (Unweighted Average Recall)
TP_C_total = [x for x in TP_C_total if x != 0]
TP_total = [x for x in TP_total if x != 0]
print("UAR:",sum([float(x) / float(y)
                    for x, y in zip(TP_total, TP_C_total)])/5)
# WAR
print("WAR:",sum(TP_total) / sum(TP_C_total))
confusion_mat=confusion_matrix(true_label,pred_label)
col_sum = confusion_mat.sum(axis=0)
# print('running_loss_t:', running_loss_t)
# sort the model list as the order of loss and keep the top 5
# modellist = [x for _, x in sorted(zip(running_loss_t, modellist))]
# # average the running_loss
# running_loss_t = [sum(running_loss_t) / len(running_loss_t)]
# # add new row to dataframe
# result_file.loc[len(modellist)] = ["ALL",running_loss_t ,sum(TP_total) / sum(TP_C_total), TP_C_total, TP_total]
# # save the result to csv
# result_file.to_csv('result1_file.csv', index=False)

# ...
# Print the confusion matrix to the console
print(confusion_mat)
# convert to percentage with 2 decimal and 100%
confusion_mat = np.around(confusion_mat*100 / np.sum(confusion_mat, axis=1)[:, np.newaxis], decimals=2)

print(confusion_mat)
# Visualize the confusion matrix as a heatmap and trans color to more clear
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(confusion_mat, cmap=plt.cm.Blues)

# Add axis labels and tick marks
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
# delete the number 0 in TP_C_total
TP_C_total = [x for x in TP_C_total if x != 0]
# add total numnber of each classes to classes name
classes_pred = classes
# true
classes = [x + '\n' + str(y) for x, y in zip(classes, TP_C_total)]
ax.set_yticklabels(classes)
# predicted
classes_pred = [x + '\n' + str(y) for x, y in zip(classes_pred, col_sum)]
ax.set_xticklabels(classes_pred)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)

# Add title and axis labels
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
# add total numnber of each classes

# Loop over data to add text annotations
for i in range(len(classes)):
    for j in range(len(classes)):
        text = ax.text(j, i, confusion_mat[i, j],
                       ha="center", va="center", color="black")

# Show the plot
plt.show()