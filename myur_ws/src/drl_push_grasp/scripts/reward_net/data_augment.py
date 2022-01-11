# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import numpy as np 
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms, utils


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        # rebuild __init__() func
        super(MyDataset, self).__init__()
        fd = open(txt, 'r')
        imgs = []
        for line in fd:
            line = line.strip('\n')
            words = line.split()
            imgs.append((words[0], words[1], int(words[2])))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader


    def __getitem__(self, index):
        # rebuild __getitem__() func
        fn0, fn1, label = self.imgs[index]
        img0 = self.loader(fn0)
        img1 = self.loader(fn1)
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, label


    def __len__(self):
        return len(self.imgs)


