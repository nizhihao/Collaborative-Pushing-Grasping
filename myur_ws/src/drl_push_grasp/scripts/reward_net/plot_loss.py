# !/usr/bin/env python
# -*- coding:UTF-8 -*-
import os
from matplotlib import pyplot as plt 


def read_loss(file_name):
    with open(file_name, 'r') as fd:
        loss = fd.readlines()
    loss = [float(i.rstrip()) for i in loss]
    return loss


def loss_plot(loss):
    loss_len = len(loss)
    plt.plot(range(loss_len), loss)
    plt.show()


file_name = '/home/user/myur_ws/src/drl_push_grasp/scripts/reward_net/datasets/loss.log.txt'
loss = read_loss(file_name)
loss_plot(loss)