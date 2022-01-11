#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
###########################################################
multi_model.py: the network of the pushing-grasping method
###########################################################
'''
from collections import OrderedDict
import numpy as np
from scipy import ndimage
import torch  
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
import time
import argparse, os

device_ids = [0,1]
os.environ["CUDA_VISIBLE_DEVICES"] = '1,0'


class push_net(nn.Module):
    def __init__(self, use_cuda = True):
        super(push_net, self).__init__()
        self.use_cuda = use_cuda

        self.push_color_trunk = torchvision.models.densenet.densenet121(pretrained=True)
        self.push_depth_trunk = torchvision.models.densenet.densenet121(pretrained=True)
        self.num_rotations = 16

        # densenet feature dimension is 1024, concate color and depth channel is 2048
        self.pushnet = nn.Sequential(OrderedDict([
            ('push-norm0', nn.BatchNorm2d(2048)),
            ('push-relu0', nn.ReLU(inplace=True)),
            ('push-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('push-norm1', nn.BatchNorm2d(64)),
            ('push-relu1', nn.ReLU(inplace=True)),
            ('push-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
        ]))

        # Initialize network weights
        for m in self.named_modules():
            if 'push-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal_(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

        # Initialize output variable (for backprop)
        self.push_interm_feat = []
        self.push_output_prob = []


    def forward(self, input_color_data, input_depth_data, is_volatile=False, specific_rotation=-1):
        if is_volatile:
            with torch.no_grad():
                push_output_prob = []
                push_interm_feat = []

                # Apply rotations to images
                for rotate_idx in range(self.num_rotations):
                    rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))
                    
                    # Compute sample grid for rotation BEFORE neural network
                    affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                    affine_mat_before.shape = (2,3,1)
                    affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
                    if self.use_cuda:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_color_data.size())
                    else:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False), input_color_data.size())
                    
                    # Rotate images clockwise
                    if self.use_cuda:
                        rotate_color = F.grid_sample(Variable(input_color_data).cuda(), flow_grid_before, mode='nearest')
                        rotate_depth = F.grid_sample(Variable(input_depth_data).cuda(), flow_grid_before, mode='nearest')
                        
                    else:
                        rotate_color = F.grid_sample(Variable(input_color_data), flow_grid_before, mode='nearest')
                        rotate_depth = F.grid_sample(Variable(input_depth_data), flow_grid_before, mode='nearest')

                    # Compute intermediate features
                    interm_push_color_feat = self.push_color_trunk.features(rotate_color)
                    interm_push_depth_feat = self.push_depth_trunk.features(rotate_depth)
                    interm_push_feat = torch.cat((interm_push_color_feat, interm_push_depth_feat), dim=1)

                    # Compute sample grid for rotation AFTER branches
                    affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],[-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                    affine_mat_after.shape = (2,3,1)
                    affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float()
                    if self.use_cuda:
                        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), interm_push_feat.data.size())
                    else:
                        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False), interm_push_feat.data.size())
                    
                    push_output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.pushnet(interm_push_feat), flow_grid_after, mode='nearest'))]) #,align_corners=True

            return push_output_prob, interm_push_feat

        else:
            self.push_output_prob = []
            self.push_interm_feat = []

            # Apply rotations to intermediate features
            rotate_idx = specific_rotation     #-1
            rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))
            
            # Compute sample grid for rotation BEFORE branches
            affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
            affine_mat_before.shape = (2,3,1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
            if self.use_cuda:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_color_data.size())
            else:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False), input_color_data.size())
             
            # Rotate images clockwise
            if self.use_cuda:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False).cuda(), flow_grid_before, mode='nearest')
                rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False).cuda(), flow_grid_before, mode='nearest')
            else:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False), flow_grid_before, mode='nearest')
                rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False), flow_grid_before, mode='nearest')

            # Compute intermediate features
            interm_push_color_feat = self.push_color_trunk.features(rotate_color)
            interm_push_depth_feat = self.push_depth_trunk.features(rotate_depth)
            interm_push_feat = torch.cat((interm_push_color_feat, interm_push_depth_feat), dim=1)

            
            # Compute sample grid for rotation AFTER branches
            affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],[-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2,3,1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float()
            if self.use_cuda:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), interm_push_feat.data.size())
            else:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False), interm_push_feat.data.size())
            
            self.push_output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.pushnet(interm_push_feat), flow_grid_after, mode='nearest'))])
            return self.push_output_prob, interm_push_feat


class grasp_net(nn.Module):
    def __init__(self, use_cuda = True): # , snapshot=None
        super(grasp_net, self).__init__()
        self.use_cuda = use_cuda

        self.grasp_color_trunk = torchvision.models.densenet.densenet121(pretrained=True)
        self.grasp_depth_trunk = torchvision.models.densenet.densenet121(pretrained=True) #device=device_ids[0,1]

        self.num_rotations = 16

        self.graspnet = nn.Sequential(OrderedDict([
            ('grasp-norm0', nn.BatchNorm2d(2048)),
            ('grasp-relu0', nn.ReLU(inplace=True)),
            ('grasp-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('grasp-norm1', nn.BatchNorm2d(64)),
            ('grasp-relu1', nn.ReLU(inplace=True)),
            ('grasp-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
        ]))

        # Initialize network weights
        for m in self.named_modules():
            if 'grasp-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal_(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

        # Initialize output variable (for backprop)
        self.grasp_interm_feat = []
        self.grasp_output_prob = []


    def forward(self, input_color_data, input_depth_data, is_volatile=False, specific_rotation=-1):
        if is_volatile:
            with torch.no_grad():
                grasp_output_prob = []
                grasp_interm_feat = []

                # Apply rotations to images
                for rotate_idx in range(self.num_rotations):
                    rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))
                    
                    # Compute sample grid for rotation BEFORE neural network
                    affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                    affine_mat_before.shape = (2,3,1)
                    affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
                    if self.use_cuda:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_color_data.size())
                    else:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False), input_color_data.size())
                    
                    # Rotate images clockwise
                    if self.use_cuda:
                        rotate_color = F.grid_sample(Variable(input_color_data).cuda(), flow_grid_before, mode='nearest')
                        rotate_depth = F.grid_sample(Variable(input_depth_data).cuda(), flow_grid_before, mode='nearest')
                    else:
                        rotate_color = F.grid_sample(Variable(input_color_data), flow_grid_before, mode='nearest')
                        rotate_depth = F.grid_sample(Variable(input_depth_data), flow_grid_before, mode='nearest')

                    # Compute intermediate features
                    interm_grasp_color_feat = self.grasp_color_trunk.features(rotate_color)
                    interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
                    interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)

                    # Compute sample grid for rotation AFTER branches
                    affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],[-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                    affine_mat_after.shape = (2,3,1)
                    affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float()
                    if self.use_cuda:
                        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), interm_grasp_feat.data.size())
                    else:
                        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False), interm_grasp_feat.data.size())
                    
                    grasp_output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.graspnet(interm_grasp_feat), flow_grid_after, mode='nearest'))]) #,align_corners=True

            return grasp_output_prob, interm_grasp_feat

        else:
            self.grasp_interm_feat = []
            self.grasp_output_prob = []

            # Apply rotations to intermediate features
            rotate_idx = specific_rotation     #-1
            rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))
            
            # Compute sample grid for rotation BEFORE branches
            affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
            affine_mat_before.shape = (2,3,1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
            if self.use_cuda:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_color_data.size())
            else:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False), input_color_data.size())
             
            # Rotate images clockwise
            if self.use_cuda:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False).cuda(), flow_grid_before, mode='nearest')
                rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False).cuda(), flow_grid_before, mode='nearest')
            else:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False), flow_grid_before, mode='nearest')
                rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False), flow_grid_before, mode='nearest')

            # Compute intermediate features
            interm_grasp_color_feat = self.grasp_color_trunk.features(rotate_color)
            interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
            interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)

            # Compute sample grid for rotation AFTER branches
            affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],[-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2,3,1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float()
            if self.use_cuda:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), interm_grasp_feat.data.size())
            else:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False), interm_grasp_feat.data.size())
            
            self.grasp_output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.graspnet(interm_grasp_feat), flow_grid_after, mode='nearest'))])
            return self.grasp_output_prob, interm_grasp_feat

