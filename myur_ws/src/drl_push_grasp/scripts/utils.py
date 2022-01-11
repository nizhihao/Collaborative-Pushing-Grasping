#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
#####################################################################################################################################
utils.py: there are some function module, such as the func     def get_heightmap()    to get the color-state-map and depth-state-map
#####################################################################################################################################
'''
import struct
import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
from matplotlib import pyplot as plt
import random
import json
import time
from scipy import ndimage


def torch_to_numpy(torch_tensor, is_standardized_image=False):
    """ Converts torch tensor (NCHW) to numpy tensor (NHWC) for plotting
        If it's an rgb image, it puts it back in [0,255] range (and undoes ImageNet standardization)
    """
    np_tensor = torch_tensor.cpu().clone().detach().numpy()
    if np_tensor.ndim == 4:  # NCHW
        np_tensor = np_tensor.transpose(0, 2, 3, 1)
    if is_standardized_image:
        _mean = [0.485, 0.456, 0.406]
        _std = [0.229, 0.224, 0.225]
        for i in range(3):
            np_tensor[..., i] *= _std[i]
            np_tensor[..., i] += _mean[i]
        np_tensor *= 255

    return np_tensor


def get_pointcloud(color_img, depth_img, camera_intrinsics):
    # Get depth image size
    im_h = depth_img.shape[0]  #（480，640）
    im_w = depth_img.shape[1]

    # Project depth into 3D point cloud in camera coordinates
    pix_x,pix_y = np.meshgrid(np.linspace(0,im_w-1,im_w), np.linspace(0,im_h-1,im_h))
    cam_pts_x = np.multiply(pix_x-camera_intrinsics[0][2],depth_img/camera_intrinsics[0][0])
    cam_pts_y = np.multiply(pix_y-camera_intrinsics[1][2],depth_img/camera_intrinsics[1][1])
    cam_pts_z = depth_img.copy()
    cam_pts_x.shape = (im_h*im_w,1)
    cam_pts_y.shape = (im_h*im_w,1)
    cam_pts_z.shape = (im_h*im_w,1)

    # Reshape image into colors for 3D point cloud
    rgb_pts_r = color_img[:,:,0]
    rgb_pts_g = color_img[:,:,1]
    rgb_pts_b = color_img[:,:,2]
    rgb_pts_r.shape = (im_h*im_w,1)
    rgb_pts_g.shape = (im_h*im_w,1)
    rgb_pts_b.shape = (im_h*im_w,1)

    cam_pts = np.concatenate((cam_pts_x, cam_pts_y, cam_pts_z), axis=1)
    rgb_pts = np.concatenate((rgb_pts_r, rgb_pts_g, rgb_pts_b), axis=1)
    return cam_pts, rgb_pts


def get_heightmap(color_img, depth_img, cam_intrinsics, cam_pose, workspace_limits, heightmap_resolution, is_sim, median_filter_pixels=0):
    # get the color-state-map and the depth-state-map
    if median_filter_pixels > 0:
        depth_img = ndimage.median_filter(depth_img, size=median_filter_pixels)

    # Compute heightmap size
    heightmap_size = np.round(((workspace_limits[1][1] - workspace_limits[1][0])/heightmap_resolution, (workspace_limits[0][1] - workspace_limits[0][0])/heightmap_resolution)).astype(int)

    # Get 3D point cloud from RGB-D images
    surface_pts, color_pts = get_pointcloud(color_img, depth_img, cam_intrinsics)

    # Transform 3D point cloud from camera coordinates to robot coordinates
    surface_pts = np.transpose(np.dot(cam_pose[0:3,0:3],np.transpose(surface_pts)) + np.tile(cam_pose[0:3,3:],(1,surface_pts.shape[0])))

    # Sort surface points by z value
    sort_z_ind = np.argsort(surface_pts[:,2])
    surface_pts = surface_pts[sort_z_ind]
    color_pts = color_pts[sort_z_ind]     

    # Filter out surface points outside heightmap boundaries
    heightmap_valid_ind = np.logical_and(np.logical_and(np.logical_and(np.logical_and(surface_pts[:,0] >= workspace_limits[0][0], surface_pts[:,0] < workspace_limits[0][1]), surface_pts[:,1] >= workspace_limits[1][0]), surface_pts[:,1] < workspace_limits[1][1]), surface_pts[:,2] < workspace_limits[2][1])
    surface_pts = surface_pts[heightmap_valid_ind]
    color_pts = color_pts[heightmap_valid_ind]

    # Create orthographic top-down-view RGB-D heightmaps
    color_heightmap_r = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    color_heightmap_g = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    color_heightmap_b = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    if is_sim:
        depth_heightmap = np.ones(heightmap_size) * 0.605   # desk height(Change me)
    else:
        depth_heightmap = np.ones(heightmap_size) * 0.250   # desk height(Change me)

    heightmap_pix_x = np.floor((surface_pts[:,0] - workspace_limits[0][0])/heightmap_resolution).astype(int)
    heightmap_pix_y = np.floor((surface_pts[:,1] - workspace_limits[1][0])/heightmap_resolution).astype(int)

    color_heightmap_r[heightmap_pix_y,heightmap_pix_x] = color_pts[:,[0]]
    color_heightmap_g[heightmap_pix_y,heightmap_pix_x] = color_pts[:,[1]]
    color_heightmap_b[heightmap_pix_y,heightmap_pix_x] = color_pts[:,[2]]

    if median_filter_pixels > 0:
        color_heightmap_r = ndimage.median_filter(color_heightmap_r, size=median_filter_pixels)
        color_heightmap_g = ndimage.median_filter(color_heightmap_g, size=median_filter_pixels)
        color_heightmap_b = ndimage.median_filter(color_heightmap_b, size=median_filter_pixels)

    color_heightmap = np.concatenate((color_heightmap_r, color_heightmap_g, color_heightmap_b), axis=2)
    depth_heightmap[heightmap_pix_y,heightmap_pix_x] = surface_pts[:,2]

    if not is_sim:
        z_bottom = workspace_limits[2][0]
        depth_heightmap[depth_heightmap < z_bottom] = z_bottom

    return color_heightmap, depth_heightmap


def get_affordance_vis(grasp_affordances, input_images, num_rotations, best_pix_ind):
    # visualize the pixel-wise prediction
    vis = None
    for vis_row in range(num_rotations/4):
        tmp_row_vis = None
        for vis_col in range(4):
            rotate_idx = vis_row*4+vis_col
            affordance_vis = grasp_affordances[rotate_idx,:,:]
            affordance_vis[affordance_vis < 0] = 0 # assume probability
            # affordance_vis = np.divide(affordance_vis, np.max(affordance_vis))
            affordance_vis[affordance_vis > 1] = 1 # assume probability
            affordance_vis.shape = (grasp_affordances.shape[1], grasp_affordances.shape[2])
            affordance_vis = cv2.applyColorMap((affordance_vis*255).astype(np.uint8), cv2.COLORMAP_JET)
            input_image_vis = (input_images[rotate_idx,:,:,:]*255).astype(np.uint8)
            input_image_vis = cv2.resize(input_image_vis, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            affordance_vis = (0.5*cv2.cvtColor(input_image_vis, cv2.COLOR_RGB2BGR) + 0.5*affordance_vis).astype(np.uint8)
            if rotate_idx == best_pix_ind[0]:
                affordance_vis = cv2.circle(affordance_vis, (int(best_pix_ind[2]), int(best_pix_ind[1])), 7, (0,0,255), 2)
            if tmp_row_vis is None:
                tmp_row_vis = affordance_vis
            else:
                tmp_row_vis = np.concatenate((tmp_row_vis,affordance_vis), axis=1)
        if vis is None:
            vis = tmp_row_vis
        else:
            vis = np.concatenate((vis,tmp_row_vis), axis=0)

    return vis


def preprocess(color_img):
    # image process
    color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    color_img = cv2.bilateralFilter(color_img, 3, 30, 30)
    # gamma = 0.57
    gamma = 0.65
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    color_img = cv2.LUT(color_img, lookUpTable)

    blur = cv2.GaussianBlur(color_img, (0, 0), 25)
    color_img = cv2.addWeighted(color_img, 1.5, blur, -0.5, 0)

    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    return color_img


def policynms_push(is_sim, push_predictions, num_rotations, valid_depth_heightmap):
    # policyNMS of the push policy
    if is_sim:
        push_action_nms_value = 0
        push_index = 0
        push_final_pixel = None

        for num, push_prediction in enumerate(push_predictions):
            # create mask
            push_mask = np.zeros((224, 224), np.uint8)
            push_pixel = np.unravel_index(np.argmax(push_prediction), push_prediction.shape)
            push_predicted_value = np.max(push_prediction)

            # ROI region
            push_roi = [[min(max(push_pixel[1] - 4, 0), 223), min(max(push_pixel[0] + 11, 0), 223)],
                   [min(max(push_pixel[1] + 4, 0), 223), min(max(push_pixel[0] + 20, 0), 223)]]
            push_mask = cv2.rectangle(push_mask, (push_roi[0][0], push_roi[0][1]), (push_roi[1][0], push_roi[1][1]), (255, 255, 255), -1)

            # affine transform
            push_rotate = cv2.getRotationMatrix2D((push_pixel[1], push_pixel[0]), num*360/num_rotations, 1)
            push_final = cv2.warpAffine(push_mask, push_rotate, (224, 224))

            # get the rotating mask
            push_ROI_multiply = np.zeros(push_final.shape, dtype=np.uint8)
            push_ROI_count = np.zeros(push_final.shape, dtype=np.uint8)
            push_ROI_multiply[push_final >= 100] = 255
            push_ROI_count[push_final >= 100] = 1
            push_total_pixel = np.sum(push_ROI_count)

            # run the AND operation between ROI_count and depth-state-map，calculate the pixel of obj
            push_depth_nms = np.multiply(valid_depth_heightmap, push_ROI_count)
            push_nms_count = np.zeros(push_depth_nms.shape, dtype=np.uint8)
            push_nms_count[push_depth_nms >= 0.618] = 1
            push_obj_pixel = np.sum(push_nms_count)

            # calculate the prob
            push_prob = float(push_obj_pixel) / (float(push_total_pixel) + 1)
            push_temp = push_prob * push_predicted_value

            # get max predition and index
            if push_temp >= push_action_nms_value:
                push_action_nms_value = push_temp
                push_index = num
                push_final_pixel = push_pixel

        if push_action_nms_value == 0:
            push_locate_part = np.unravel_index(np.argmax(push_predictions), push_predictions.shape)
            push_predicted_value = np.max(push_predictions)

        else:
            push_locate_part = tuple([push_index, push_final_pixel[0], push_final_pixel[1]])
            push_predicted_value = push_action_nms_value
    else:
        push_action_nms_value = 0
        push_index = 0
        push_final_pixel = None

        for num, push_prediction in enumerate(push_predictions):

            # create mask
            push_mask = np.zeros((224, 224), np.uint8)
            push_pixel = np.unravel_index(np.argmax(push_prediction), push_prediction.shape)
            push_predicted_value = np.max(push_prediction)

            # ROI region
            push_roi = [[min(max(push_pixel[1] - 4, 0), 223), min(max(push_pixel[0] + 6, 0), 223)],
                        [min(max(push_pixel[1] + 4, 0), 223), min(max(push_pixel[0] + 16, 0), 223)]]
            push_mask = cv2.rectangle(push_mask, (push_roi[0][0], push_roi[0][1]),
                                      (push_roi[1][0], push_roi[1][1]), (255, 255, 255), -1)

            # affine transform
            push_rotate = cv2.getRotationMatrix2D((push_pixel[1], push_pixel[0]), num*360/num_rotations + 180, 1)
            push_final = cv2.warpAffine(push_mask, push_rotate, (224, 224))

            # get the rotating mask
            push_ROI_multiply = np.zeros(push_final.shape, dtype=np.uint8)
            push_ROI_count = np.zeros(push_final.shape, dtype=np.uint8)
            push_ROI_multiply[push_final >= 100] = 255
            push_ROI_count[push_final >= 100] = 1
            push_total_pixel = np.sum(push_ROI_count)

            # run the AND operation between ROI_count and depth-state-map，calculate the pixel of obj
            push_depth_nms = np.multiply(valid_depth_heightmap, push_ROI_count)
            push_nms_count = np.zeros(push_depth_nms.shape, dtype=np.uint8)
            push_nms_count[push_depth_nms >= 0.272] = 1
            push_obj_pixel = np.sum(push_nms_count)

            # calculate the prob
            if push_total_pixel < 30:
                push_prob = 0.01
            else:
                push_prob = float(push_obj_pixel) / (float(push_total_pixel) + 1)
            push_temp = push_prob * push_predicted_value

            # get max predition and index
            if push_temp >= push_action_nms_value:
                push_action_nms_value = push_temp
                push_index = num
                push_final_pixel = push_pixel

        if push_action_nms_value == 0:
            push_locate_part = np.unravel_index(np.argmax(push_predictions), push_predictions.shape)
            push_predicted_value = np.max(push_predictions)

        else:
            push_locate_part = tuple([push_index, push_final_pixel[0], push_final_pixel[1]])
            push_predicted_value = push_action_nms_value
    
    return push_locate_part, push_predicted_value


def policynms_grasp(is_sim, grasp_predictions, num_rotations, valid_depth_heightmap):
    # policyNMS of the grasp policy
    if is_sim:
        grasp_action_nms_value = 0
        grasp_index = 0
        grasp_final_pixel = None

        for num, grasp_prediction in enumerate(grasp_predictions):
            # create mask
            mask = np.zeros((224, 224), np.uint8)
            mask1 = np.zeros((224, 224), np.uint8)
            pixel = np.unravel_index(np.argmax(grasp_prediction), grasp_prediction.shape)
            grasp_predicted_value = np.max(grasp_prediction)

            # ROI region
            roi = [[min(max(pixel[1] - 6, 0), 223), min(max(pixel[0] - 16, 0), 223)],
                   [min(max(pixel[1] + 6, 0), 223), min(max(pixel[0] - 10, 0), 223)]]

            roi1 = [[min(max(pixel[1] - 6, 0), 223), min(max(pixel[0] + 10, 0), 223)],
                   [min(max(pixel[1] + 6, 0), 223), min(max(pixel[0] + 16, 0), 223)]]
            mask = cv2.rectangle(mask, (roi[0][0], roi[0][1]), (roi[1][0], roi[1][1]),
                                 (255, 255, 255), -1)

            mask1 = cv2.rectangle(mask1, (roi1[0][0], roi1[0][1]), (roi1[1][0], roi1[1][1]),
                                 (255, 255, 255), -1)

            # affine transform
            rotate = cv2.getRotationMatrix2D((pixel[1], pixel[0]), (num * 180 / num_rotations), 1)
            final = cv2.warpAffine(mask, rotate, (224, 224))
            final1 = cv2.warpAffine(mask1, rotate, (224, 224))

            # get the rotating mask
            ROI_multiply = np.zeros(final.shape, dtype=np.uint8)
            ROI_count = np.zeros(final.shape, dtype=np.uint8)
            ROI_multiply[final >= 100] = 255
            ROI_count[final >= 100] = 1
            total_pixel = np.sum(ROI_count)

            ROI_multiply1 = np.zeros(final1.shape, dtype=np.uint8)
            ROI_count1 = np.zeros(final1.shape, dtype=np.uint8)
            ROI_multiply1[final1 >= 100] = 255
            ROI_count1[final1 >= 100] = 1
            total_pixel1 = np.sum(ROI_count1)

            # run the AND operation between ROI_count and depth-state-map，calculate the pixel of obj
            depth_nms = np.multiply(valid_depth_heightmap, ROI_count)
            nms_count = np.zeros(depth_nms.shape, dtype=np.uint8)
            nms_count[depth_nms >= 0.618] = 1
            obj_pixel = np.sum(nms_count)

            depth_nms1 = np.multiply(valid_depth_heightmap, ROI_count1)
            nms_count1 = np.zeros(depth_nms1.shape, dtype=np.uint8)
            nms_count1[depth_nms1 >= 0.618] = 1
            obj_pixel1 = np.sum(nms_count1)

            # calculate the prob
            prob = float(obj_pixel) / (float(total_pixel) + 1)
            prob1 = float(obj_pixel1) / (float(total_pixel1) + 1)
            final_prob = 1 - max(prob, prob1)
            temp = final_prob * grasp_predicted_value

            # get max predition and index
            if temp >= grasp_action_nms_value:
                grasp_action_nms_value = temp
                grasp_index = num
                grasp_final_pixel = pixel

        if grasp_action_nms_value == 0:
            locate_part = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
            grasp_predicted_value = np.max(grasp_predictions)

        else:
            locate_part = tuple([grasp_index, grasp_final_pixel[0], grasp_final_pixel[1]])
            grasp_predicted_value = grasp_action_nms_value
    else:
        grasp_action_nms_value = 0
        grasp_index = 0
        grasp_final_pixel = None

        for num, grasp_prediction in enumerate(grasp_predictions):
            # create mask
            mask = np.zeros((224, 224), np.uint8)
            mask1 = np.zeros((224, 224), np.uint8)
            pixel = np.unravel_index(np.argmax(grasp_prediction), grasp_prediction.shape)
            grasp_predicted_value = np.max(grasp_prediction)

            # ROI region
            roi = [[min(max(pixel[1] - 7, 0), 223), min(max(pixel[0] - 17, 0), 223)],
                   [min(max(pixel[1] + 7, 0), 223), min(max(pixel[0] - 10, 0), 223)]]

            roi1 = [[min(max(pixel[1] - 7, 0), 223), min(max(pixel[0] + 10, 0), 223)],
                   [min(max(pixel[1] + 7, 0), 223), min(max(pixel[0] + 17, 0), 223)]]
            mask = cv2.rectangle(mask, (roi[0][0], roi[0][1]), (roi[1][0], roi[1][1]),
                                 (255, 255, 255), -1)

            mask1 = cv2.rectangle(mask1, (roi1[0][0], roi1[0][1]), (roi1[1][0], roi1[1][1]),
                                 (255, 255, 255), -1)

            # affine transform
            rotate = cv2.getRotationMatrix2D((pixel[1], pixel[0]), (num * 180 / num_rotations + 90), 1)
            final = cv2.warpAffine(mask, rotate, (224, 224))
            final1 = cv2.warpAffine(mask1, rotate, (224, 224))

            # get the rotating mask
            ROI_multiply = np.zeros(final.shape, dtype=np.uint8)
            ROI_count = np.zeros(final.shape, dtype=np.uint8)
            ROI_multiply[final >= 100] = 255
            ROI_count[final >= 100] = 1
            total_pixel = np.sum(ROI_count)

            ROI_multiply1 = np.zeros(final1.shape, dtype=np.uint8)
            ROI_count1 = np.zeros(final1.shape, dtype=np.uint8)
            ROI_multiply1[final1 >= 100] = 255
            ROI_count1[final1 >= 100] = 1
            total_pixel1 = np.sum(ROI_count1)

            # run the AND operation between ROI_count and depth-state-map，calculate the pixel of obj
            depth_nms = np.multiply(valid_depth_heightmap, ROI_count)
            nms_count = np.zeros(depth_nms.shape, dtype=np.uint8)
            nms_count[depth_nms >= 0.270] = 1
            obj_pixel = np.sum(nms_count)

            depth_nms1 = np.multiply(valid_depth_heightmap, ROI_count1)
            nms_count1 = np.zeros(depth_nms1.shape, dtype=np.uint8)
            nms_count1[depth_nms1 >= 0.270] = 1
            obj_pixel1 = np.sum(nms_count1)

            # calculate the prob
            prob = float(obj_pixel) / (float(total_pixel) + 1)
            prob1 = float(obj_pixel1) / (float(total_pixel1) + 1)
            final_prob = 1 - max(prob, prob1)
            temp = final_prob * grasp_predicted_value

            # get max predition and index
            if temp >= grasp_action_nms_value:
                grasp_action_nms_value = temp
                grasp_index = num
                grasp_final_pixel = pixel

        if grasp_action_nms_value == 0:
            locate_part = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
            grasp_predicted_value = np.max(grasp_predictions)

        else:
            locate_part = tuple([grasp_index, grasp_final_pixel[0], grasp_final_pixel[1]])
            grasp_predicted_value = grasp_action_nms_value

    return locate_part, grasp_predicted_value


def maxpushHeight(best_pix_x, best_pix_y, depth_img, best_rotation_angle):
    # push_height
    detect_height_list = []
    for i in range(max(0, best_pix_x-1), min(224, best_pix_x+2)):
        for j in range(max(0, best_pix_y-1),min(224, best_pix_y+2)):
            detect_height_list.append(depth_img[j][i])

    sort_height_list = sorted(detect_height_list, reverse=True)
    max_z = 0
    n = 1
    for i in range(n):
        max_z += sort_height_list[i]
    average_push_z = float(max_z / n)

    return average_push_z


def maxgraspHeight_sim(best_pix_x, best_pix_y, valid_depth_heightmap, num):
    # grasp_height, such as policyNMS
    mask = np.zeros((224, 224), np.uint8)
    mask1 = np.zeros((224, 224), np.uint8)

    roi = [[min(max(best_pix_x - 7, 0), 223), min(max(best_pix_y - 16, 0), 223)],  #16 13
           [min(max(best_pix_x + 7, 0), 223), min(max(best_pix_y - 14, 0), 223)]]

    roi1 = [[min(max(best_pix_x - 7, 0), 223), min(max(best_pix_y + 14, 0), 223)],
            [min(max(best_pix_x + 7, 0), 223), min(max(best_pix_y + 16, 0), 223)]]
    mask = cv2.rectangle(mask, (roi[0][0], roi[0][1]), (roi[1][0], roi[1][1]),
                         (255, 255, 255), -1)

    mask1 = cv2.rectangle(mask1, (roi1[0][0], roi1[0][1]), (roi1[1][0], roi1[1][1]),
                          (255, 255, 255), -1)

    rotate = cv2.getRotationMatrix2D((best_pix_x, best_pix_y), (num * 180 / 16), 1)
    final = cv2.warpAffine(mask, rotate, (224, 224))
    final1 = cv2.warpAffine(mask1, rotate, (224, 224))

    ROI_multiply = np.zeros(final.shape, dtype=np.uint8)
    ROI_count = np.zeros(final.shape, dtype=np.uint8)
    ROI_multiply[final >= 100] = 255
    ROI_count[final >= 100] = 1

    ROI_multiply1 = np.zeros(final1.shape, dtype=np.uint8)
    ROI_count1 = np.zeros(final1.shape, dtype=np.uint8)
    ROI_multiply1[final1 >= 100] = 255
    ROI_count1[final1 >= 100] = 1

    depth_nms = np.multiply(valid_depth_heightmap, ROI_count)
    depth_nms1 = np.multiply(valid_depth_heightmap, ROI_count1)

    max_z = max(np.max(depth_nms), np.max(depth_nms1))
    return max_z


def maxgraspHeight(best_pix_x, best_pix_y, depth_img, best_rotation_angle):
    # average height of grasp center
    cent_detect_height_list = []
    for i in range(max(0, best_pix_x-1), min(224, best_pix_x+2)):
        for j in range(max(0, best_pix_y-1),min(224, best_pix_y+2)):
            distance2Cent = ((left_cent_x - i) ** 2 + (best_pix_y - j) ** 2) ** 0.5
            diff_x = i - best_pix_x
            diff_y = j - best_pix_y
            if diff_x == 0:
                if diff_y < 0:
                    rotateAng = - np.pi/2
                elif diff_y > 0:
                    rotateAng = np.pi/2
                else:
                    rotateAng = 0
            else:
                rotateAng =  math.atan((diff_y / diff_x))

            y = min(max(0, int(np.round(distance2Cent * np.sin(rotateAng + tool_rotation_angle) + best_pix_y))), 223)
            x = min(max(0, int(np.round(distance2Cent * np.cos(rotateAng + tool_rotation_angle) + best_pix_x))), 223)
            cent_detect_height_list.append(depth_img[y][x])

    cent_sort_height_list = sorted(cent_detect_height_list, reverse=True)
    cent_maxz = (cent_sort_height_list[1] + cent_sort_height_list[2] + cent_sort_height_list[3] + cent_sort_height_list[4]) / 4

    return cent_maxz



def euler2rotm(theta):
    # Get rotation matrix from euler angles
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])         
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])            
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R


def save_data_heightmap(color_heightmap, depth_heightmap, iteration, state):
    # save the reward-net's data
    color_heightmap = cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2BGR)
    save_img_file = '/home/user/reward_dataset/IMG_%06d/'%(iteration)
    if not os.path.exists(save_img_file):
        os.makedirs(save_img_file)
    cv2.imwrite(os.path.join(save_img_file, 'color%s.png' % (state)), color_heightmap)
    depth_heightmap = np.round(depth_heightmap * 104470).astype(np.uint16)    # 1E-5   104470    237432
    cv2.imwrite(os.path.join(save_img_file, 'depth%s.png' % (state)), depth_heightmap)


def write_to_data_log(log_name, iteration, log):
    # save the reward-net's label
    save_param_file = '/home/user/reward_dataset/IMG_%06d/' % (iteration)
    if not os.path.exists(save_param_file):
        os.makedirs(save_param_file)
    np.savetxt(os.path.join(save_param_file, '%s.txt' % log_name), log, delimiter=' ')


