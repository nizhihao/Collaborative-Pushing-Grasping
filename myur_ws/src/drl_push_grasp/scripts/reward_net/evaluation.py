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

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
BATCH_SIZE = 32
EPOCH = 20

save_path = '/home/user/myur_ws/src/drl_push_grasp/scripts/reward_net/datasets/'
test_txt = '/home/user/myur_ws/src/drl_push_grasp/scripts/reward_net/datasets/depth_test.txt'
model_param = '/home/user/myur_ws/src/drl_push_grasp/scripts/reward_net/datasets/params/reward_net_transfer_2cls_0.pth'

train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_data = MyDataset(txt=test_txt, transform=train_transforms)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

use_cuda = True
if use_cuda:    
    model = reward_net(use_cuda).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
else:
    model = reward_net(use_cuda)
    criterion = nn.CrossEntropyLoss()

model.load_state_dict(torch.load(model_param))

# testing
for epoch in range(EPOCH):
    print('{}/{}'.format(epoch + 1, EPOCH))
    print('*'*15)
    print('Evaluation')
    model.eval()  

    eval_loss = 0.0
    running_acc = 0.0
    total = 0.0

    with torch.no_grad():
        for i, data in enumerate(test_loader, 1):
            img0, img1, label = data
            if use_cuda:
                label = label.cuda() 

            # FORWARD 
            prediction = model(img0, img1)
            loss = criterion(prediction, label)
            _, pred = torch.max(prediction, 1) 

            eval_loss += loss.data * label.size(0)  
            num_correct = torch.sum(pred == label)
            running_acc += num_correct.data
            total += label.size(0)

        print('Loss: {:.6f}, Acc: {:.4f}'.format(eval_loss / total, float(running_acc) / total))

print('Eval Done!!')



