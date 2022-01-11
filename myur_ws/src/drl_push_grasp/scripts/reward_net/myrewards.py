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
import numpy as np
from model import reward_net
import cv2
from data_augment import MyDataset
import util

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
MODEL_PARAM = '/home/user/myur_ws/src/drl_push_grasp/scripts/reward_net/datasets/params/reward_net_transfer_2cls_0.pth'


class MyRewards(object):
    def __init__(self, use_cuda):
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model = reward_net(self.use_cuda).cuda()
        else:
            self.model = reward_net(self.use_cuda)
        self.model.load_state_dict(torch.load(MODEL_PARAM))


    def get_reward(self, prev_color_heightmap, cur_color_heightmap):
        self.model.eval()
        prev_input, cur_input = self.normalize(prev_color_heightmap, cur_color_heightmap)
        with torch.no_grad():
            if self.use_cuda:
                # input shape = (1, 3, 224, 224)
                prev_input.shape = (prev_input.shape[0], prev_input.shape[1], prev_input.shape[2], 1)
                cur_input.shape = (cur_input.shape[0], cur_input.shape[1], cur_input.shape[2], 1)
                input_before_data = torch.from_numpy(prev_input.astype(np.float32)).permute(3, 2, 0, 1)
                input_cur_data = torch.from_numpy(cur_input.astype(np.float32)).permute(3, 2, 0, 1)

            prediction = self.model(input_before_data, input_cur_data)
            _, pred = torch.max(prediction, 1)

            # reproject
            if pred == 0:
                my_reward = -0.5
            elif pred == 1:
                my_reward = 0.5

        return my_reward

    def normalize(self, prev_color_heightmap, cur_color_heightmap):
        image_mean = [0.5, 0.5, 0.5]
        image_std = [0.5, 0.5, 0.5]
        before_color_heightmap = prev_color_heightmap.astype(float) / 255
        cur_color_heightmap = cur_color_heightmap.astype(float) / 255
        for c in range(3):
            before_color_heightmap[:, :, c] = (before_color_heightmap[:, :, c] - image_mean[c]) / image_std[c]
        for c in range(3):
            cur_color_heightmap[:, :, c] = (cur_color_heightmap[:, :, c] - image_mean[c]) / image_std[c]

        return before_color_heightmap, cur_color_heightmap

