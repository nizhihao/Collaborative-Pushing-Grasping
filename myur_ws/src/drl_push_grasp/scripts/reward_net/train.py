# !/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import division
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils

from model import reward_net
from data_augment import MyDataset
import util

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

BATCH_SIZE = 32
LR = 1e-6
EPOCH = 50

# dataload
save_path = '/home/user/myur_ws/src/drl_push_grasp/scripts/reward_net/datasets/'
train_txt = '/home/user/myur_ws/src/drl_push_grasp/scripts/reward_net/atasets/depth_train.txt'
model_path = '/home/user/myur_ws/src/drl_push_grasp/scripts/reward_net/datasets/params/reward_net_transfer_2cls_0.pth'
model_param = '/home/user/myur_ws/src/drl_push_grasp/scripts/reward_net/datasets/params/'

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = MyDataset(txt=train_txt, transform=train_transforms)
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# Define model, optimizer, loss 
use_cuda = True
if use_cuda:    
    model = reward_net(use_cuda).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
else:
    model = reward_net(use_cuda)
    criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.01)
loss_record = []

# load weight
model.load_state_dict(torch.load(model_path))

# train
for epoch in range(EPOCH):
    print('{}/{}'.format(epoch + 1, EPOCH))
    print('*'*15)
    print('Train')
    model.train()  

    running_loss = 0.0
    running_acc = 0.0
    since = time.time()

    for i, data in enumerate(train_loader, 1):
        img0, img1, label = data

        if use_cuda:
            label = label.cuda() 

        # FORWARD 
        prediction = model(img0, img1)
        loss = criterion(prediction, label)
        _, pred = torch.max(prediction, 1) 

        # save each iteration loss value
        loss_record.append(loss)
        util.write_to_log('loss', loss_record, save_path)

        # BACKWARD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   

        running_loss += loss.data * label.size(0)  
        num_correct = torch.sum(pred == label)
        running_acc += num_correct.data
        if i % 100 == 0:
            print('Loss: {:.6f}, Acc: {:.4f}'.format(running_loss / float(i * BATCH_SIZE), float(running_acc) / float(i * BATCH_SIZE)))

    running_loss = float(running_loss) / float(len(train_loader.dataset))
    running_acc = float(running_acc) / float(len(train_loader.dataset))
    elips_time = time.time() - since
    print('Loss: {:.6f}, Acc: {:.4f}, Time: {:.0f}s'.format(running_loss, running_acc, elips_time))

    # save
    util.save_backup_model(model, model_param)

print('DONE!!')




