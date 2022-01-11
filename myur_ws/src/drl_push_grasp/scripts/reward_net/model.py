# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import torchvision
from collections import OrderedDict


class reward_net(nn.Module):
    def __init__(self, use_cuda=True):
        super(reward_net, self).__init__()
        self.use_cuda = use_cuda
        
        self.before_trunk = torchvision.models.vgg16(pretrained=True)
        self.after_trunk = torchvision.models.vgg16(pretrained=True)

        self.fusion_layer = nn.Sequential(OrderedDict([
            ('fusion-norm0', nn.BatchNorm2d(1024)),
            ('fusion-relu0', nn.ReLU(inplace=True)),
            ('fusion-conv0', nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False)),
            ('fusion-norm1', nn.BatchNorm2d(512)),
            ('fusion-relu1', nn.ReLU(inplace=True)),
            ('fusion-conv1', nn.Conv2d(512, 512, kernel_size=3, stride=1, bias=False))
        ]))

        self.full_connect_layer = nn.Sequential(OrderedDict([
            ('connect-linear0', nn.Linear(5*5*512, 2048)),
            ('connect-relu0', nn.ReLU(inplace=True)),
            ('connect-dropout0', nn.Dropout(p=0.5)),
            ('connect-linear1', nn.Linear(2048, 512)),
            ('connect-relu1', nn.ReLU(inplace=True)),
            ('connect-dropout1', nn.Dropout(p=0.5)),
            ('connect-linear2', nn.Linear(512, 2))
        ]))


        # init param
        for m in self.named_modules():
            if 'fusion-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal_(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()


    def forward(self, color_before_imgs, color_after_imgs):
        '''
        input: 
            color_before_imgs: size(BxCxHxW)=(64,3,224,224)
            color_after_imgs: size(BxCxHxW)=(64,3,224,224)
        
        output:
            reward:size(B,1)=(64,1)
        '''
        if self.use_cuda:
            input_before = Variable(color_before_imgs).cuda()
            input_after = Variable(color_after_imgs).cuda()
        else:
            input_before = Variable(color_before_imgs)
            input_after = Variable(color_after_imgs)

        trunk_before = self.before_trunk.features(input_before)
        trunk_after = self.after_trunk.features(input_after)
        fusion = torch.cat((trunk_before, trunk_after), dim=1)
        fusion_out = self.fusion_layer(fusion)
        fullconnect_in = fusion_out.view(fusion_out.size(0), -1)
        n_class = self.full_connect_layer(fullconnect_in)

        return n_class

