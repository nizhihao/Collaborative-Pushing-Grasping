#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
##########################################################
trainer.py: agent module，forward/backward
##########################################################
'''
import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from multi_model import push_net, grasp_net
from scipy import ndimage
import matplotlib.pyplot as plt
from reward_net import myrewards
import utils

target_angle = 0

class Trainer(object):
    def __init__(self, push_rewards, future_reward_discount,
                 is_testing, load_snapshot, push_snapshot_file, grasp_snapshot_file, force_cpu, is_sim):
        self.is_sim = is_sim
        self.prediction_reward = myrewards.MyRewards(use_cuda=True)

        # Check if CUDA can be used
        if torch.cuda.is_available() and not force_cpu:
            print("CUDA detected. Running with GPU acceleration.")
            self.use_cuda = True
        elif force_cpu:
            print("CUDA detected, but overriding with option '--cpu'. Running with only CPU.")
            self.use_cuda = False
        else:
            print("CUDA is *NOT* detected. Running with only CPU.")
            self.use_cuda = False

        # Fully convolutional Q network for deep reinforcement learning
        self.push_model = push_net(self.use_cuda)
        self.grasp_model = grasp_net(self.use_cuda)
        self.push_rewards = push_rewards
        self.future_reward_discount = future_reward_discount

        # Initialize Huber loss
        self.push_criterion = torch.nn.SmoothL1Loss(reduce=False)
        self.grasp_criterion = torch.nn.SmoothL1Loss(reduce=False)
        if self.use_cuda:
            self.push_criterion = self.push_criterion.cuda()
            self.grasp_criterion = self.grasp_criterion.cuda()

        # Load pre-trained model
        if load_snapshot:
            self.push_model.load_state_dict(torch.load(push_snapshot_file))
            self.grasp_model.load_state_dict(torch.load(grasp_snapshot_file))
            print('Pre-trained model snapshot loaded from: %s' % (push_snapshot_file))

        # Convert model from CPU to GPU
        if self.use_cuda:
            self.push_model = self.push_model.cuda()
            self.grasp_model = self.grasp_model.cuda()
        
        # Set model to training mode
        self.push_model.train()
        self.grasp_model.train()

        # Initialize optimizer
        self.push_optimizer = torch.optim.SGD(self.push_model.parameters(), lr=1e-4, momentum=0.9, weight_decay=2e-5)
        self.grasp_optimizer = torch.optim.SGD(self.grasp_model.parameters(), lr=1e-4, momentum=0.9, weight_decay=2e-5)
        self.iteration = 0

        # Initialize lists to save execution info and RL variables
        self.executed_action_log = []
        self.label_value_log = []
        self.reward_value_log = []
        self.predicted_value_log = []
        self.use_heuristic_log = []
        self.is_exploit_log = []
        self.clearance_log = []
        self.success_log = []
        self.pushout_log = []


    def preload(self, transitions_directory):
        # Pre-load execution info and RL variables
        self.executed_action_log = np.loadtxt(os.path.join(transitions_directory, 'executed-action.log.txt'), delimiter=' ')
        self.iteration = self.executed_action_log.shape[0] - 2
        self.executed_action_log = self.executed_action_log[0:self.iteration,:]
        self.executed_action_log = self.executed_action_log.tolist()
        self.label_value_log = np.loadtxt(os.path.join(transitions_directory, 'label-value.log.txt'), delimiter=' ')
        self.label_value_log = self.label_value_log[0:self.iteration]
        self.label_value_log.shape = (self.iteration,1)
        self.label_value_log = self.label_value_log.tolist()
        self.predicted_value_log = np.loadtxt(os.path.join(transitions_directory, 'predicted-value.log.txt'), delimiter=' ')
        self.predicted_value_log = self.predicted_value_log[0:self.iteration]
        self.predicted_value_log.shape = (self.iteration,1)
        self.predicted_value_log = self.predicted_value_log.tolist()
        self.reward_value_log = np.loadtxt(os.path.join(transitions_directory, 'reward-value.log.txt'), delimiter=' ')
        self.reward_value_log = self.reward_value_log[0:self.iteration]
        self.reward_value_log.shape = (self.iteration,1)
        self.reward_value_log = self.reward_value_log.tolist()
        self.use_heuristic_log = np.loadtxt(os.path.join(transitions_directory, 'use-heuristic.log.txt'), delimiter=' ')
        self.use_heuristic_log = self.use_heuristic_log[0:self.iteration]
        self.use_heuristic_log.shape = (self.iteration,1)
        self.use_heuristic_log = self.use_heuristic_log.tolist()
        self.is_exploit_log = np.loadtxt(os.path.join(transitions_directory, 'is-exploit.log.txt'), delimiter=' ')
        self.is_exploit_log = self.is_exploit_log[0:self.iteration]
        self.is_exploit_log.shape = (self.iteration,1)
        self.is_exploit_log = self.is_exploit_log.tolist()
        self.clearance_log = np.loadtxt(os.path.join(transitions_directory, 'clearance.log.txt'), delimiter=' ')
        self.clearance_log.shape = (self.clearance_log.shape[0],1)
        self.clearance_log = self.clearance_log.tolist()

        self.success_log = np.loadtxt(os.path.join(transitions_directory, 'success.log.txt'), delimiter=' ')
        self.success_log.shape = (self.success_log.shape[0],2)
        self.success_log = self.success_log.tolist()

        self.pushout_log = np.loadtxt(os.path.join(transitions_directory, 'pushout.log.txt'), delimiter=' ')
        self.pushout_log.shape = (self.pushout_log.shape[0],1)
        self.pushout_log = self.pushout_log.tolist()


    def forward(self, color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=-1):
        # Compute forward pass through model to compute affordances/Q
        color_heightmap_2x = ndimage.zoom(color_heightmap, zoom=[2,2,1], order=0)
        depth_heightmap_2x = ndimage.zoom(depth_heightmap, zoom=[2,2], order=0)
        assert(color_heightmap_2x.shape[0:2] == depth_heightmap_2x.shape[0:2])

        # Add extra padding (to handle rotations inside network) 
        diag_length = float(color_heightmap_2x.shape[0]) * np.sqrt(2)  
        diag_length = np.ceil(diag_length/32)*32 
        padding_width = int((diag_length - color_heightmap_2x.shape[0])/2)

        #pad(array, pad_width, mode, **kwargs)
        color_heightmap_2x_r = np.pad(color_heightmap_2x[:,:,0], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_r.shape = (color_heightmap_2x_r.shape[0], color_heightmap_2x_r.shape[1], 1)
        color_heightmap_2x_g = np.pad(color_heightmap_2x[:,:,1], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_g.shape = (color_heightmap_2x_g.shape[0], color_heightmap_2x_g.shape[1], 1)
        color_heightmap_2x_b = np.pad(color_heightmap_2x[:,:,2], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_b.shape = (color_heightmap_2x_b.shape[0], color_heightmap_2x_b.shape[1], 1)
        color_heightmap_2x = np.concatenate((color_heightmap_2x_r, color_heightmap_2x_g, color_heightmap_2x_b), axis=2)
        depth_heightmap_2x = np.pad(depth_heightmap_2x, padding_width, 'constant', constant_values=0)

        # Pre-process color image (scale and normalize)
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        
        input_color_image = color_heightmap_2x.astype(float)/255
        for c in range(3):
            input_color_image[:,:,c] = (input_color_image[:,:,c] - image_mean[c])/image_std[c]

        # Pre-process depth image (normalize)
        image_mean = [0.01, 0.01, 0.01]
        image_std = [0.03, 0.03, 0.03]

        cp_depth_heightmap = depth_heightmap_2x.copy()
        cp_depth_heightmap.shape = (cp_depth_heightmap.shape[0], cp_depth_heightmap.shape[1], 1)
        input_depth_image = np.concatenate((cp_depth_heightmap, cp_depth_heightmap, cp_depth_heightmap), axis=2)
        for c in range(3):
            input_depth_image[:,:,c] = (input_depth_image[:,:,c] - image_mean[c])/image_std[c]
        
        # Construct minibatch of size 1 (b,c,h,w)
        input_color_image.shape = (input_color_image.shape[0], input_color_image.shape[1], input_color_image.shape[2], 1)
        input_depth_image.shape = (input_depth_image.shape[0], input_depth_image.shape[1], input_depth_image.shape[2], 1)
        input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(3,2,0,1)
        input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(3,2,0,1)

        push_output_prob, _ = self.push_model.forward(input_color_data, input_depth_data, is_volatile, specific_rotation)
        grasp_output_prob, _ = self.grasp_model.forward(input_color_data, input_depth_data, is_volatile, specific_rotation)

        # Return Q values (and remove extra padding)
        for rotate_idx in range(len(push_output_prob)):
            if rotate_idx == 0:
                push_predictions = push_output_prob[rotate_idx][0].cpu().data.numpy()[:,0,48:272,48:272]
                grasp_predictions = grasp_output_prob[rotate_idx][0].cpu().data.numpy()[:,0,48:272,48:272]
            else:
                push_predictions = np.concatenate((push_predictions, push_output_prob[rotate_idx][0].cpu().data.numpy()[:,0,48:272,48:272]), axis=0)
                grasp_predictions = np.concatenate((grasp_predictions, grasp_output_prob[rotate_idx][0].cpu().data.numpy()[:,0,48:272,48:272]), axis=0)

        return push_predictions, grasp_predictions, None


    def get_label_value(self, primitive_action, grasp_success, change_detected, cur_color_heightmap, cur_depth_heightmap, prev_reward_depth_heightmap, reward_depth_heightmap, inside, judge_reset, Win):
        # obtain the reward, gamma * max(q(s_,A))
        Fail = False
        current_reward = -0.1
        if primitive_action == 'push':
            if self.is_sim:
                if change_detected:
                    current_reward = self.prediction_reward.get_reward(prev_reward_depth_heightmap, reward_depth_heightmap)
                    print('reward value from reward_net: {}'.format(current_reward))
                # push out the workspace 
                if not inside:
                    Fail = True
                    current_reward = -5.0
            else:
                if change_detected:
                    current_reward = self.prediction_reward.get_reward(prev_reward_depth_heightmap, reward_depth_heightmap)
                    print('reward value from reward_net: {}'.format(current_reward))

        elif primitive_action == 'grasp':
            if self.is_sim:
                if grasp_success:
                    current_reward = 1.5
                    if Win:
                        current_reward = 5.0
                        Win = False
            else:
                # in test, default Grasp = True
                if grasp_success and judge_reset == False:
                    current_reward = 1.5
                    Grasp = True
                elif grasp_success and judge_reset == True:
                    current_reward = 0.75
                    Grasp = True
                else:
                    Grasp = False

        # Compute future reward
        if not change_detected:
            future_reward = 0
        else:
            next_push_predictions, next_grasp_predictions, _ = self.forward(cur_color_heightmap, cur_depth_heightmap, is_volatile=True)
            future_reward = max(np.max(next_grasp_predictions), np.max(next_push_predictions))

        print('Current reward: %f' % (current_reward))
        print('Future reward: %f' % (future_reward))

        if primitive_action == 'push' and not self.push_rewards:
            expected_reward = self.future_reward_discount * future_reward
            print('Expected reward: %f + %f x %f = %f' % (0.0, self.future_reward_discount, future_reward, expected_reward))
        else:
            expected_reward = current_reward + self.future_reward_discount * future_reward
            print('Expected reward: %f + %f x %f = %f' % (current_reward, self.future_reward_discount, future_reward, expected_reward))

        return expected_reward, current_reward, Fail, Win


    def backprop(self, color_heightmap, depth_heightmap, primitive_action, best_pix_ind, label_value):
        # Compute labels and backpropagate
        label = np.zeros((1,320,320))
        action_area = np.zeros((224,224))
        action_area[best_pix_ind[1]][best_pix_ind[2]] = 1
        tmp_label = np.zeros((224,224))
        tmp_label[action_area > 0] = label_value
        label[0,48:(320-48),48:(320-48)] = tmp_label

        # Compute label mask
        label_weights = np.zeros(label.shape)
        tmp_label_weights = np.zeros((224,224))
        tmp_label_weights[action_area > 0] = 1
        label_weights[0,48:(320-48),48:(320-48)] = tmp_label_weights
        
        # Compute loss and backward pass
        self.push_optimizer.zero_grad()
        self.grasp_optimizer.zero_grad()
        loss_value = 0
        if primitive_action == 'push':
            # Do forward pass with specified rotation (to save gradients)
            push_predictions, grasp_predictions, _ = self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0])

            if self.use_cuda:
                loss = self.push_criterion(self.push_model.push_output_prob[0][0].view(1,320,320), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
            else:
                loss = self.push_criterion(self.push_model.push_output_prob[0][0].view(1,320,320), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(),requires_grad=False)

            push_loss = loss.sum()
            push_loss.backward(retain_graph=True)
            try:
                loss_value = push_loss.cpu().data.numpy()[0]
            except:
                loss_value = push_loss.cpu().data.numpy()
            self.push_optimizer.step()

        elif primitive_action == 'grasp':
            # Do forward pass with specified rotation (to save gradients)
            push_predictions, grasp_predictions, state_feat = self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0])
            
            if self.use_cuda:
                grasp_loss = self.grasp_criterion(self.grasp_model.grasp_output_prob[0][0].view(1,320,320), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
            else:
                grasp_loss = self.grasp_criterion(self.grasp_model.grasp_output_prob[0][0].view(1,320,320), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(),requires_grad=False)

            grasp_loss = grasp_loss.sum()
            grasp_loss.backward(retain_graph=True)
            try:
                loss_value = grasp_loss.cpu().data.numpy()[0]
            except:
                loss_value = grasp_loss.cpu().data.numpy()

            self.grasp_optimizer.step()
        print('Training loss: %f' % (loss_value))


    def get_prediction_vis(self, predictions, color_heightmap, best_pix_ind):
        # visualize pixel-wise prediction
        canvas = None 
        num_rotations = predictions.shape[0]             #push/grasp_predictions (16,224,224)
        for canvas_row in range(int(num_rotations/4)):
            tmp_row_canvas = None
            for canvas_col in range(4):
                rotate_idx = canvas_row*4+canvas_col
                prediction_vis = predictions[rotate_idx,:,:].copy()
                prediction_vis = np.clip(prediction_vis, 0, 1)
                prediction_vis.shape = (predictions.shape[1], predictions.shape[2])
                prediction_vis = cv2.applyColorMap((prediction_vis*255).astype(np.uint8), cv2.COLORMAP_JET)
                if rotate_idx == best_pix_ind[0]:
                    prediction_vis = cv2.circle(prediction_vis, (int(best_pix_ind[2]), int(best_pix_ind[1])), 7, (0,0,255), 2)
                prediction_vis = ndimage.rotate(prediction_vis, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
                background_image = ndimage.rotate(color_heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
                prediction_vis = (0.5*cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR) + 0.5*prediction_vis).astype(np.uint8)
                if tmp_row_canvas is None:
                    tmp_row_canvas = prediction_vis
                else:
                    tmp_row_canvas = np.concatenate((tmp_row_canvas,prediction_vis), axis=1)
            if canvas is None:
                canvas = tmp_row_canvas
            else:
                canvas = np.concatenate((canvas,tmp_row_canvas), axis=0)

        return canvas


