# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch 
import numpy as np 
import cv2

def dataset2txt(datasets_path, save_path):
    # dataset file convert to path
    files = os.listdir(datasets_path)
    sort_files = sorted(files, key=lambda x:x[4:10], reverse=False)

    fd = open(os.path.join(save_path, 'depth_train.txt'), 'a')
    for file in sort_files:
        data0 = os.path.join(datasets_path, file, 'colorbefore.png')
        data1 = os.path.join(datasets_path, file, 'colorafter.png')
        with open(os.path.join(datasets_path, file, 'label.txt'), 'r') as file_object:
            label = file_object.read()

        if label >= 0.5:
            label = 1
        line = data0 + ' ' + data1 + ' ' + str(label) + '\n'
        fd.write(line)
    fd.close()


def dataset2txt_data_from_drl(datasets_path, save_path):
    # dataset file convert to path(dataset from drl interaction)
    files = os.listdir(datasets_path)
    sort_files = sorted(files, key=lambda x: x[4:], reverse=False)

    fd = open(os.path.join(save_path, 'depth_train.txt'), 'a')
    for file in sort_files:
        data0 = os.path.join(datasets_path, file, 'colorbefore.png')
        data1 = os.path.join(datasets_path, file, 'colorafter.png')
        with open(os.path.join(datasets_path, file, 'label.txt'), 'r') as file_object:
            label = file_object.read()
            label = label.rstrip()
            print(label)

        if label == '-1.000000000000000056e-01' or label == '-5.000000000000000000e-01' or label == '-5.000000000000000000e+00' or label[0] == '-':
            label = 0
        elif label == '5.000000000000000000e-01':
            label = 1
        elif label == '5.000000000000000000e+00':
            label = 2
        else:
            continue
        line = data0 + ' ' + data1 + ' ' + str(label) + '\n'
        fd.write(line)
    fd.close()


def save_model(model, path, epoch):
    torch.save(model.cpu().state_dict(), os.path.join(path, 'reward_net_%06d.pth' % epoch))


def save_backup_model(model, path):
    torch.save(model.state_dict(), os.path.join(path, 'reward_net.backup.pth'))


def write_to_log(log_name, log, save_path):
    np.savetxt(os.path.join(save_path, '%s.log.txt' % log_name), log, delimiter=' ')


def negative_datasets(datasets_path, save_path):
    # negative_data are the inverse of dataset
    files = os.listdir(datasets_path)
    sort_files = sorted(files, key=lambda x:x[4:], reverse=False)

    fd = open(os.path.join(save_path, 'depth_train.txt'), 'a')
    for file in sort_files:
        data1 = os.path.join(datasets_path, file, 'depth_before.png')
        data0 = os.path.join(datasets_path, file, 'depth_after.png')
        label = 0
        line = data0 + ' ' + data1 + ' ' + str(label) + '\n'
        fd.write(line)
    fd.close()


def cv2PIL(datasets_path, save_path):
    # save the img with dtype=int32, and input the img with PIL
    files = os.listdir(datasets_path)
    sort_files = sorted(files, key=lambda x:x[4:], reverse=False)

    for file in sort_files:
        data0 = os.path.join(datasets_path, file, 'depthbefore.png')
        src0 = cv2.imread(data0)
        data0_after = os.path.join(datasets_path, file, 'depth_before.png')
        cv2.imwrite(data0_after, src0)

        data1 = os.path.join(datasets_path, file, 'depthafter.png')
        src1 = cv2.imread(data1)
        data1_after = os.path.join(datasets_path, file, 'depth_after.png')
        cv2.imwrite(data1_after, src1)


# generate train.txt
datasets_path = '/home/user/new_drl_datasets(more_like_real)_before20201019/'
save_path = '/home/user/myur_ws/src/drl_push_grasp/scripts/reward_net/datasets/'

dataset2txt(datasets_path, save_path)
negative_datasets(datasets_path, save_path)


