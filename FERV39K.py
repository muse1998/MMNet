import math
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import cv2
import pandas as pd
import os, torch
import torch.nn as nn
#import image_utils
import argparse, random
from functools import partial
from CA_block import resnet18_pos_attention
from PC_module import VisionTransformer_POS
from torchvision.transforms import Resize
from torch.utils.tensorboard import SummaryWriter # to print to tensorboard
import warnings
# Suppress the warning message related to the antialias parameter
warnings.filterwarnings("ignore", message="The default value of the antialias parameter of all the resizing transforms")
torch.set_printoptions(precision=3, edgeitems=14, linewidth=350)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='', help='Raf-DB dataset path.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Pytorch checkpoint file path')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Pretrained weights')
    parser.add_argument('--beta', type=float, default=0.7, help='Ratio of high importance group in one mini-batch.')
    parser.add_argument('--relabel_epoch', type=int, default=1000,
                        help='Relabeling samples on each mini-batch after 10(Default) epochs.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=7000, help='Total training epochs.')
    parser.add_argument('--drop_rate', type=float, default=0, help='Drop out rate.')
    return parser.parse_args()






class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, transform = None, basic_aug = False, transform_norm=None):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path
        self.transform_norm = transform_norm
        NAME_COLUMN = 0
        APEX_COLUMN = 1
        OFF_COLUMN = 2
        LABEL_ALL_COLUMN = 3

        if phase == 'train':
            dataset = pd.read_csv("result_ferv39k.csv",usecols=[0,1,2,3])
        else:
            dataset = pd.read_csv("t_result_ferv39k.csv",usecols=[0,1,2,3])

        File_names = dataset.iloc[:, NAME_COLUMN].values
        Label_all = dataset.iloc[:, LABEL_ALL_COLUMN].values  # 0:Happy, 1:Sad, 2:Neutral, 3:Anger, 4:Surprise, 5:Disgust, 6:fear
        Apex_num = dataset.iloc[:, APEX_COLUMN].values
        Offset_num = dataset.iloc[:, OFF_COLUMN].values
        self.file_paths_on = []
        self.file_paths_off = []
        self.file_paths_apex = []
        self.label_all = []
        self.file_names =[]
        print(Label_all)
        # a=0
        # b=0
        # c=0
        # d=0
        # e=0
        # use aligned images for training/testing
        for (f,apex,offset,label_all) in zip(File_names,Apex_num,Offset_num,Label_all):
            if label_all < 7:
                self.file_paths_on.append(0)
                self.file_paths_off.append(offset-1)
                self.file_paths_apex.append(apex)
                self.file_names.append(f)
                if label_all == 0:
                    self.label_all.append(0)
                elif label_all == 6:
                    self.label_all.append(1)
                elif label_all == 1:
                    self.label_all.append(2)
                elif label_all == 5:
                    self.label_all.append(3)
                elif label_all == 3:
                    self.label_all.append(4)                    
                elif label_all == 4:
                    self.label_all.append(5)
                else:
                    self.label_all.append(6)
                    


            ##label

        self.basic_aug = basic_aug
        #self.aug_func = [image_utils.flip_image,image_utils.add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths_on)

    def __getitem__(self, idx):
        ##sampling strategy for training set
        if self.phase == 'train':
            onset = self.file_paths_on[idx]
            apex = self.file_paths_apex[idx]
            offset =self.file_paths_off[idx]
            on0 = random.randint(int(onset), int(onset + int(0.15* (apex - onset) / 4)))
            # on0 = str(int(onset))
            on1 = random.randint(int(onset + int(0.9 * (apex - onset) / 4)), int(onset + int(1.1 * (apex - onset) / 4)))
            on2 = random.randint(int(onset + int(1.8 * (apex - onset) / 4)), int(onset + int(2.2 * (apex - onset) / 4)))
            on3 = random.randint(int(onset + int(2.7 * (apex - onset) / 4)), onset + int(3.3 * (apex - onset) / 4))
            # apex0 = str(apex)
            apex0 = random.randint(int(apex - int(0.15* (apex - onset) / 4)), apex)
            off0 = random.randint(int(apex + int(0.9 * (offset - apex) / 4)), int(apex + int(1.1 * (offset - apex) / 4)))
            off1 = random.randint(int(apex + int(1.8 * (offset - apex) / 4)), int(apex + int(2.2 * (offset - apex) / 4)))
            off2 = random.randint(int(apex + int(2.9 * (offset - apex) / 4)), int(apex + int(3.1 * (offset - apex) / 4)))
            off3 = random.randint(int(apex + int(3.8 * (offset - apex) / 4)), offset)
            f = str(self.file_names[idx])
        else:##sampling strategy for testing set
            onset = self.file_paths_on[idx]
            apex = self.file_paths_apex[idx]
            offset = self.file_paths_off[idx]

            on0 = (onset)
            on1 = (int(onset + int((apex - onset) / 4)))
            on2 = (int(onset + int(2 * (apex - onset) / 4)))
            on3 = (int(onset + int(3 * (apex - onset) / 4)))
            apex0 = (apex)
            off0 = (int(apex + int((offset - apex) / 4)))
            off1 = (int(apex + int(2 * (offset - apex) / 4)))
            off2 = (int(apex + int(3 * (offset - apex) / 4)))
            off3 = (offset)

            f = str(self.file_names[idx])

        img_list=os.listdir(os.path.join(f))
        # sort only without the last 4 characters
        img_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f[:-4]))))
        path_on0 = os.path.join(f, img_list[on0])
        path_on1 = os.path.join(f, img_list[on1])
        path_on2 = os.path.join(f, img_list[on2])
        path_on3 = os.path.join(f, img_list[on3])
        path_apex0 = os.path.join(f, img_list[apex0])
        path_off0 = os.path.join(f, img_list[off0])
        path_off1 = os.path.join(f, img_list[off1])
        path_off2 = os.path.join(f, img_list[off2])
        path_off3 = os.path.join(f, img_list[off3])
        image_on0 = cv2.imread(path_on0)
        image_on1= cv2.imread(path_on1)
        image_on2 = cv2.imread(path_on2)
        image_on3 = cv2.imread(path_on3)
        image_apex0 = cv2.imread(path_apex0)
        image_off0 = cv2.imread(path_off0)
        image_off1 = cv2.imread(path_off1)
        image_off2 = cv2.imread(path_off2)
        image_off3 = cv2.imread(path_off3)

        image_on0 = image_on0[:, :, ::-1] # BGR to RGB
        image_on1 = image_on1[:, :, ::-1]
        image_on2 = image_on2[:, :, ::-1]
        image_on3 = image_on3[:, :, ::-1]
        image_off0 = image_off0[:, :, ::-1]
        image_off1 = image_off1[:, :, ::-1]
        image_off2 = image_off2[:, :, ::-1]
        image_off3 = image_off3[:, :, ::-1]
        image_apex0 = image_apex0[:, :, ::-1]

        label_all = self.label_all[idx]

        # normalization for testing and training
        if self.transform is not None:
            image_on0 = self.transform(image_on0)
            image_on1 = self.transform(image_on1)
            image_on2 = self.transform(image_on2)
            image_on3 = self.transform(image_on3)
            image_off0 = self.transform(image_off0)
            image_off1 = self.transform(image_off1)
            image_off2 = self.transform(image_off2)
            image_off3 = self.transform(image_off3)
            image_apex0 = self.transform(image_apex0)
            ALL = torch.cat(
                (image_on0, image_on1, image_on2, image_on3, image_apex0, image_off0, image_off1, image_off2,
                 image_off3), dim=0)
            ## data augmentation for training only
            if self.transform_norm is not None and self.phase == 'train':
                ALL = self.transform_norm(ALL)
            image_on0 = ALL[0:3, :, :]
            image_on1 = ALL[3:6, :, :]
            image_on2 = ALL[6:9, :, :]
            image_on3 = ALL[9:12, :, :]
            image_apex0 = ALL[12:15, :, :]
            image_off0 = ALL[15:18, :, :]
            image_off1 = ALL[18:21, :, :]
            image_off2 = ALL[21:24, :, :]
            image_off3 = ALL[24:27, :, :]


            temp = torch.zeros(65)
        # # trans image to cuda
        # image_on0 = image_on0.to(device)
        # image_on1 = image_on1.to(device)
        # image_on2 = image_on2.to(device)
        # image_on3 = image_on3.to(device)
        # image_apex0 = image_apex0.to(device)
        # image_off0 = image_off0.to(device)
        # image_off1 = image_off1.to(device)
        # image_off2 = image_off2.to(device)
        # image_off3 = image_off3.to(device)
        # label_all = label_all.to(device)
        # temp = temp.to(device)

        return image_on0, image_on1, image_on2, image_on3, image_apex0, image_off0, image_off1, image_off2, image_off3, label_all, temp


def initialize_weight_goog(m, n=''):
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)  # fan-out
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        m.weight.data.uniform_(-init_range, init_range)
        m.bias.data.zero_()


def criterion2(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat((y_pred_neg, zeros), dim=-1)
    y_pred_pos = torch.cat((y_pred_pos, zeros), dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return torch.mean(neg_loss + pos_loss)


class MMNet(nn.Module):
    def __init__(self):
        super(MMNet, self).__init__()


        self.conv_act = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=90*2, kernel_size=3, stride=2,padding=1, bias=False,groups=1),
            nn.BatchNorm2d(180),
            nn.ReLU(inplace=True),

            )
        self.pos =nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            )
        ##Position Calibration Module(subbranch)
        self.vit_pos=VisionTransformer_POS(img_size=14,
        patch_size=1, embed_dim=512, depth=2, num_heads=4, mlp_ratio=2, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6),drop_path_rate=0.)
        self.resize=Resize([14,14])
        ##main branch consisting of CA blocks
        self.main_branch =resnet18_pos_attention()

        self.head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1 * 112 *112, 65,bias=False),

        )

        self.timeembed = nn.Parameter(torch.zeros(1, 4, 111, 111))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, if_shuffle):
        ##onset:x1 apex:x5
        B = x1.shape[0]
        # Position Calibration Module (subbranch)
        POS =self.vit_pos(self.resize(x1)).transpose(1,2).view(B,512,14,14)

        act =x5 -x1
        act=self.conv_act(act)
        # main branch and fusion
        out,_=self.main_branch(act,POS)

        return out





def run_training():
    args = parse_args()
    imagenet_pretrained = True

    if not imagenet_pretrained:
        for m in res18.modules():
            initialize_weight_goog(m)

    if args.pretrained:
        print("Loading pretrained weights...", args.pretrained)
        pretrained = torch.load(args.pretrained)
        pretrained_state_dict = pretrained['state_dict']
        model_state_dict = res18.state_dict()
        loaded_keys = 0
        total_keys = 0
        for key in pretrained_state_dict:
            if ((key == 'module.fc.weight') | (key == 'module.fc.bias')):
                pass
            else:
                model_state_dict[key] = pretrained_state_dict[key]
                total_keys += 1
                if key in model_state_dict:
                    loaded_keys += 1
        print("Loaded params num:", loaded_keys)
        print("Total params num:", total_keys)
        res18.load_state_dict(model_state_dict, strict=False)
    ##data normalization for both training set
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),

    ])
    ### data augmentation for training set only
    data_transforms_norm = transforms.Compose([

        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(3),
        transforms.RandomCrop(224, padding=15),

    ])

    ### data normalization for both teating set
    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])




    criterion = torch.nn.CrossEntropyLoss()
    # leave one subject out protocal
    # 'spNO.1', 'spNO.2', 'spNO.3', 'spNO.5', 'spNO.6', 'spNO.7', 'spNO.8', 'spNO.9', 'spNO.10', 'spNO.11', 'spNO.12', 'spNO.13', 'spNO.14', 'spNO.15', 'spNO.17', 'spNO.39', 'spNO.40', 'spNO.41', 'spNO.42', 'spNO.77', 'spNO.138', 'spNO.139', 'spNO.142', 'spNO.143', 'spNO.144', 'spNO.145', 'spNO.146', 'spNO.147', 'spNO.148', 'spNO.149', 'spNO.150', 'spNO.152', 'spNO.153', 'spNO.154', 'spNO.155', 'spNO.157', 'spNO.159', 'spNO.160', 'spNO.161', 'spNO.162', 'spNO.163', 'spNO.165', 'spNO.166', 'spNO.167', 'spNO.168', 'spNO.169', 'spNO.170', 'spNO.171', 'spNO.172', 'spNO.173', 'spNO.174', 'spNO.175', 'spNO.176', 'spNO.177', 'spNO.178', 'spNO.179', 'spNO.180', 'spNO.181', 'spNO.182', 'spNO.183', 'spNO.184', 'spNO.185', 'spNO.186', 'spNO.187', 'spNO.188', 'spNO.189', 'spNO.190', 'spNO.192', 'spNO.194', 'spNO.195', 'spNO.196', 'spNO.197', 'spNO.198', 'spNO.200', 'spNO.201', 'spNO.202', 'spNO.203', 'spNO.204', 'spNO.206', 'spNO.207', 'spNO.208', 'spNO.209', 'spNO.210', 'spNO.211', 'spNO.212', 'spNO.213', 'spNO.214', 'spNO.216', 'spNO.217'
    val_now = 0
    num_sum = 0
    pos_pred_ALL = torch.zeros(7)
    pos_label_ALL = torch.zeros(7)
    TP_ALL = torch.zeros(7)

    
    writer = SummaryWriter('./log/casme3')
    train_dataset = RafDataSet(args.raf_path, phase='train', transform=data_transforms, basic_aug=True, transform_norm=data_transforms_norm)
    val_dataset = RafDataSet(args.raf_path, phase='test',  transform=data_transforms_val)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=64,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               prefetch_factor=2,
                                               persistent_workers=True) 
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=64,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True,
                                             prefetch_factor=2,
                                             persistent_workers=True)
    
    print('Train set size:', train_dataset.__len__())
    print('Validation set size:', val_dataset.__len__())
    max_acc = 0
    max_corr = 0
    max_f1 = 0
    max_pos_pred = torch.zeros(7)
    max_pos_label = torch.zeros(7)
    max_TP = torch.zeros(7)
    ##model initialization
    net_all = MMNet()
    net_all=net_all.to(device)
    params_all = net_all.parameters()

    if args.optimizer == 'adam':
        optimizer_all = torch.optim.AdamW(params_all, lr=0.0008, weight_decay=0.6)
        ##optimizer for MMNet

    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=1e-4)
    else:
        raise ValueError("Optimizer not supported.")
    ##lr_decay
    scheduler_all = torch.optim.lr_scheduler.ExponentialLR(optimizer_all, gamma=0.987)


    for i in range(1, 75):
        running_loss = 0.0
        correct_sum = 0
        running_loss_MASK = 0.
        correct_sum_MASK = 0
        iter_cnt = 0
        net_all.train()

        for batch_i, (
        image_on0, image_on1, image_on2, image_on3, image_apex0, image_off0, image_off1, image_off2, image_off3,
        label_all,
        label_au) in enumerate(train_loader):
            batch_sz = image_on0.size(0)
            b, c, h, w = image_on0.shape
            iter_cnt += 1

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

            ##train MMNet
            ALL = net_all(image_on0, image_on1, image_on2, image_on3, image_apex0, image_off0, image_off1,
                          image_off2, image_off3, False)

            loss_all = criterion(ALL, label_all)
            optimizer_all.zero_grad()
            loss_all.backward()
            optimizer_all.step()
            running_loss += loss_all
            _, predicts = torch.max(ALL, 1)
            correct_num = torch.eq(predicts, label_all).sum()
            correct_sum += correct_num
        writer.add_scalar('Train/Loss', running_loss / iter_cnt, i)
        writer.add_scalar('Train/Acc', correct_sum.float() / float(train_dataset.__len__()), i)
        ## lr decay
        if i <= 50:
            scheduler_all.step()
        if i>=0:
            acc = correct_sum.float() / float(train_dataset.__len__())
            running_loss = running_loss / iter_cnt

            print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f' % (i, acc, running_loss))
            if(acc>=max_acc):
                max_acc=acc
                torch.save(net_all.state_dict(),
                                'model_FERV39K.pth')
                print('model saved')
        pos_label = torch.zeros(7)
        pos_pred = torch.zeros(7)
        TP = torch.zeros(7)
        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            pre_lab_all = []
            Y_test_all = []
            net_all.eval()
            # net_au.eval()
            for batch_i, (
                image_on0, image_on1, image_on2, image_on3, image_apex0, image_off0, image_off1, image_off2,
                image_off3, label_all,
            label_au) in enumerate(val_loader):
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
                    ##test
                ALL = net_all(image_on0, image_on1, image_on2, image_on3, image_apex0, image_off0, image_off1, image_off2, image_off3, False)


                loss = criterion(ALL, label_all)
                running_loss += loss
                iter_cnt += 1
                _, predicts = torch.max(ALL, 1)
                correct_num = torch.eq(predicts, label_all)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += ALL.size(0)

                for cls in range(7):
                    for element in predicts:
                        if element == cls:
                            pos_label[cls] = pos_label[cls] + 1
                    for element in label_all:
                        if element == cls:
                            pos_pred[cls] = pos_pred[cls] + 1
                    for elementp, elementl in zip(predicts, label_all):
                        if elementp == elementl and elementp == cls:
                            TP[cls] = TP[cls] + 1
                        # if pos_label != 0 or pos_pred != 0:
                        #     f1 = 2 * TP / (pos_pred + pos_label)
                        #     F1.append(f1)
                count = 0
                SUM_F1 = 0
                for index in range(7):
                    if pos_label[index] != 0 or pos_pred[index] != 0:
                        count = count + 1
                        SUM_F1 = SUM_F1 + 2 * TP[index] / (pos_pred[index] + pos_label[index])

                AVG_F1 = SUM_F1 / count


            running_loss = running_loss / iter_cnt
            acc = bingo_cnt.float() / float(sample_cnt)
            acc = np.around(acc.numpy(), 4)
            if bingo_cnt > max_corr:
                max_corr = bingo_cnt
            if AVG_F1 >= max_f1:
                max_f1 = AVG_F1
                max_pos_label = pos_label
                max_pos_pred = pos_pred
                max_TP = TP
            writer.add_scalar('Val/Loss', running_loss, i)
            writer.add_scalar('Val/Acc', acc, i)
            print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f, F1-score:%.3f" % (i, acc, running_loss, AVG_F1))
    num_sum = num_sum + max_corr
    pos_label_ALL = pos_label_ALL + max_pos_label
    pos_pred_ALL = pos_pred_ALL + max_pos_pred
    TP_ALL = TP_ALL + max_TP
    count = 0
    SUM_F1 = 0
    for index in range(7):
        if pos_label_ALL[index] != 0 or pos_pred_ALL[index] != 0:
            count = count + 1
            SUM_F1 = SUM_F1 + 2 * TP_ALL[index] / (pos_pred_ALL[index] + pos_label_ALL[index])
    F1_ALL = SUM_F1 / count
    val_now = val_now + val_dataset.__len__()
    print("[..........%s] correctnum:%d . zongshu:%d   " % (subj, max_corr, val_dataset.__len__()))
    print("[ALL_corr]: %d [ALL_val]: %d" % (num_sum, val_now))
    print("[F1_now]: %.4f [F1_ALL]: %.4f" % (max_f1, F1_ALL))



if __name__ == "__main__":
    run_training()
