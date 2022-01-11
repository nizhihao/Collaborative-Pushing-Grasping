#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
######################################################################################
main.py: this code is the final version of the collaborative pushing-grasping method, 
         which load the pre-train push/grasp policy to train the collaborative param
         or to test the performance.
######################################################################################
'''
import time
import os
import random
import threading
import argparse
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import cv2
from collections import namedtuple
import torch  
from torch.autograd import Variable
from trainer import Trainer
from logger import Logger
import utils
from ur_robotiq import Ur10


def main(args):
    # --------------- Setup options ---------------
    is_sim = args.is_sim
    random_seed = args.random_seed
    force_cpu = args.force_cpu

    if is_sim:
        workspace_limits = np.asarray([[0.358, 1.142], [-0.392, 0.392], [0.60, 1]])
        heightmap_resolution = 0.0035
    else:
        workspace_limits = np.asarray([[-0.416, 0.256], [0.338, 1.01], [0.250, 0.51]])
        heightmap_resolution = 0.003

    # ------------- Algorithm options -------------
    push_rewards = args.push_rewards
    future_reward_discount = args.future_reward_discount
    experience_replay = args.experience_replay
    heuristic_bootstrap = args.heuristic_bootstrap
    explore_rate_decay = args.explore_rate_decay

    # -------------- Testing options --------------
    is_testing = args.is_testing
    max_test_trials = args.max_test_trials

    # ------ Pre-loading and logging options ------
    load_snapshot = args.load_snapshot

    # multi_model
    if load_snapshot: 
        push_snapshot_file, grasp_snapshot_file = os.path.abspath(args.push_snapshot_file), os.path.abspath(args.grasp_snapshot_file)  
    else:
        push_snapshot_file, grasp_snapshot_file = None, None
        
    continue_logging = args.continue_logging
    logging_directory = os.path.abspath(args.logging_directory) if continue_logging else os.path.abspath('/home/user/myur_ws/src/drl_push_grasp/logs')  # 放到服务器上的时候修改一下
    save_visualizations = args.save_visualizations
    np.random.seed(random_seed)

    # interact nevironment
    robot = Ur10(is_sim, is_testing, workspace_limits)

    # agnet model
    trainer = Trainer(push_rewards, future_reward_discount, is_testing, load_snapshot, push_snapshot_file, grasp_snapshot_file, force_cpu, is_sim)

    # Initialize data logger
    logger = Logger(continue_logging, logging_directory)
    logger.save_camera_info(robot.cam_intrinsics, robot.cam_pose, robot.cam_depth_scale) # Save camera intrinsics and pose
    logger.save_heightmap_info(workspace_limits, heightmap_resolution) # Save heightmap parameters

    # Find last executed iteration of pre-loaded log, and load execution info and RL variables
    if continue_logging:
        trainer.preload(logger.transitions_directory)

    # Initialize variables
    no_change_count = [2, 2] if not is_testing else [0, 0]
    explore_prob = 0.4 if not is_testing else 0.0
    judge_reset = False
    num_rotations = 16
    Fail = False
    Win = False
    epoch_count = 1

    # Quick hack for nonlocal memory between threads in Python 2
    nonlocal_variables = {'executing_action' : False,
                          'primitive_action' : None,
                          'best_push_pix_ind' : None,
                          'best_grasp_pix_ind': None,
                          'best_pix_ind': None,
                          'push_success' : False,
                          'grasp_success' : False}


    def process_actions():
        # Parallel thread to process network output and execute actions
        while True:
            if nonlocal_variables['executing_action']:
                # obtain the max push prediction
                nonlocal_variables['best_push_pix_ind'], push_predicted_value = utils.policynms_push(is_sim, push_predictions, num_rotations, valid_depth_heightmap)

                # obtain the max grasp prediction
                nonlocal_variables['best_grasp_pix_ind'], grasp_predicted_value = utils.policynms_grasp(is_sim, grasp_predictions, num_rotations, valid_depth_heightmap)

                # compare the push and grasp prediction
                print('Primitive confidence scores: %f (push), %f (grasp)' % (push_predicted_value, grasp_predicted_value))
                if push_predicted_value > grasp_predicted_value:
                    nonlocal_variables['primitive_action'] = 'push'
                    nonlocal_variables['best_pix_ind'] = nonlocal_variables['best_push_pix_ind']
                    predicted_value = push_predicted_value
                else:
                    nonlocal_variables['primitive_action'] = 'grasp'
                    nonlocal_variables['best_pix_ind'] = nonlocal_variables['best_grasp_pix_ind']
                    predicted_value = grasp_predicted_value

                # explore
                if np.random.uniform() < explore_prob:
                    explore_action_index = np.random.choice(range(2))
                    if explore_action_index == 0:
                        nonlocal_variables['primitive_action'] = 'grasp'
                        nonlocal_variables['best_pix_ind'] = nonlocal_variables['best_grasp_pix_ind']
                        predicted_value = grasp_predicted_value
                        print('Strategy: explore grasp(exploration probability: %f)' % (explore_prob))
                    else:
                        nonlocal_variables['primitive_action'] = 'push'
                        nonlocal_variables['best_pix_ind'] = nonlocal_variables['best_push_pix_ind']
                        predicted_value = push_predicted_value
                        print('Strategy: explore push(exploration probability: %f)' % (explore_prob))
                else:
                    print('Strategy: exploit (exploration probability: %f)' % (explore_prob))

                # Save predicted confidence value
                trainer.predicted_value_log.append([predicted_value])
                logger.write_to_log('predicted-value', trainer.predicted_value_log)

                # Compute 3D position of pixel，[0] - rotate，[1] - height，[2] - width
                print('Action: %s at (%d, %d, %d)' % (nonlocal_variables['primitive_action'], nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]))
                best_rotation_angle = np.deg2rad(nonlocal_variables['best_pix_ind'][0]*(360.0/num_rotations))
                best_pix_x = nonlocal_variables['best_pix_ind'][2]
                best_pix_y = nonlocal_variables['best_pix_ind'][1]

                diff_hegiht = 0
                cur_height = valid_depth_heightmap[best_pix_y][best_pix_x]
                print("Current world z-axis hegiht: {}".format(cur_height))

                if is_sim:
                    # safe region in push
                    if nonlocal_variables['primitive_action'] == 'push':
                        max_push_z = utils.maxpushHeight(best_pix_x, best_pix_y, valid_depth_heightmap, best_rotation_angle)
                        down = max_push_z * (-0.8) + 0.565
                    # safe region in grasp
                    else:
                        max_grasp_z = utils.maxgraspHeight_sim(best_pix_x, best_pix_y, valid_depth_heightmap, nonlocal_variables['best_grasp_pix_ind'][0])
                        up = max_grasp_z - 0.605
                    
                    primitive_position = [best_pix_x * heightmap_resolution + workspace_limits[0][0], best_pix_y * heightmap_resolution + workspace_limits[1][0], 0.97]
                else:
                    backgroundH = 0.25  # (change me)
                    if nonlocal_variables['primitive_action'] == 'push':
                        max_push_z = utils.maxpushHeight(best_pix_x, best_pix_y, valid_depth_heightmap, best_rotation_angle)
                        # 0.02, 0.012 is offset,  + 0.025 mean the diff between end effortor and gripper
                        down = 0.45 + abs(max_push_z - backgroundH) + 0.005 + 0.025 + 0.02 - 0.012
                    
                    elif nonlocal_variables['primitive_action'] == 'grasp':
                        center_maxz = utils.maxgraspHeight(best_pix_x, best_pix_y, valid_depth_heightmap, best_rotation_angle)
                        down = max(0.48, 0.45 + abs(center_maxz - backgroundH) + 0.005 + 0.025 + 0.02 - 0.012 - 0.01)
                    
                    primitive_position = [best_pix_x * heightmap_resolution + workspace_limits[0][0], best_pix_y * heightmap_resolution + workspace_limits[1][0], 0.647]
                print('Actual Position: {}'.format(primitive_position))

                # Save executed primitive
                if nonlocal_variables['primitive_action'] == 'push':
                    trainer.executed_action_log.append([0, nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]]) # 0 - push
                elif nonlocal_variables['primitive_action'] == 'grasp':
                    trainer.executed_action_log.append([1, nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]]) # 1 - grasp
                logger.write_to_log('executed-action', trainer.executed_action_log)

                # Visualize executed primitive, and affordances
                if save_visualizations:
                    push_pred_vis = trainer.get_prediction_vis(push_predictions, color_heightmap, nonlocal_variables['best_push_pix_ind'])
                    logger.save_visualizations(trainer.iteration, push_pred_vis, 'push')
                    cv2.imwrite('visualization.push.png', push_pred_vis)
                    grasp_pred_vis = trainer.get_prediction_vis(grasp_predictions, color_heightmap, nonlocal_variables['best_grasp_pix_ind'])
                    logger.save_visualizations(trainer.iteration, grasp_pred_vis, 'grasp')
                    cv2.imwrite('visualization.grasp.png', grasp_pred_vis)

                # Initialize variables that influence reward
                nonlocal_variables['push_success'] = False
                nonlocal_variables['grasp_success'] = False
                nonlocal_variables['num_in_workspace'] = None
                change_detected = False

                # Execute primitive
                if nonlocal_variables['primitive_action'] == 'push':
                    if is_sim:
                        nonlocal_variables['push_success'] = robot.mypush(primitive_position, best_rotation_angle, workspace_limits, down)
                    else:
                        nonlocal_variables['push_success'] = robot.realpush(primitive_position, best_rotation_angle, workspace_limits, down)
                    print('Push successful: %r' % (nonlocal_variables['push_success']))
                elif nonlocal_variables['primitive_action'] == 'grasp':
                    if is_sim:
                        nonlocal_variables['grasp_success'], nonlocal_variables['num_in_workspace'] = robot.mygrasp(primitive_position, best_rotation_angle, workspace_limits, up)
                    else:
                        nonlocal_variables['grasp_success'] = robot.realgrasp(primitive_position, best_rotation_angle, workspace_limits, down)
                    print('Grasp successful: %r' % (nonlocal_variables['grasp_success']))

                nonlocal_variables['executing_action'] = False

            time.sleep(0.1)
    action_thread = threading.Thread(target=process_actions)
    action_thread.daemon = True
    action_thread.start()
    exit_called = False


    while True:
        # Start main training/testing loop
        print('\n%s iteration: %d' % ('Testing' if is_testing else 'Training', trainer.iteration))
        iteration_time_0 = time.time()
        color_img, depth_img = robot.get_camera_data()

        # img process
        if not is_sim:
            color_img = utils.preprocess(color_img)
        depth_img = depth_img * robot.cam_depth_scale
        
        # Get state-map from RGB-D image (by re-projecting 3D point cloud)
        color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, robot.cam_intrinsics, robot.cam_pose, workspace_limits, heightmap_resolution, is_sim)
        valid_depth_heightmap = depth_heightmap.copy()

        # if the obj out of the workspace, give this action a -5 reward and restart a new epoch
        if is_sim:
            inside = robot.get_env_obj_pos()
        else:
            inside = None

        # set the nan value to the height of the desk
        if is_sim: 
            valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0.605
        else:
            valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0.250
        B_distance = valid_depth_heightmap.copy()

        # Save RGB-D images and RGB-D heightmaps
        logger.save_images(trainer.iteration, color_img, depth_img, '0')
        reward_depth_heightmap = logger.save_heightmaps(trainer.iteration, color_heightmap, valid_depth_heightmap, '0')

        # Reset simulation or pause real-world training if table is empty
        stuff_count = np.zeros(B_distance.shape, dtype=np.uint8)   # stuff_count represent a mask img，dtype = np.uint8
        black_count = np.zeros(B_distance.shape)                   # black_count to count the area of the obj

        if is_sim:
            stuff_count[B_distance > 0.618] = 255
            black_count[B_distance > 0.618] = 1
            empty_threshold = 200
        else:
            stuff_count[B_distance >= 0.271] = 255
            black_count[B_distance >= 0.271] = 1
            empty_threshold = 50

        # grasp_mask
        grasp_binary_segment = stuff_count.copy()
        grasp_mor_gradient = np.zeros((224, 224), np.uint8)
        grasp_kernel = np.ones((5, 5), np.uint8)
        grasp_delete_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        grasp_mor_segment = cv2.morphologyEx(grasp_binary_segment, cv2.MORPH_OPEN, grasp_delete_small)

        if is_sim:
            grasp_mor_dilate = cv2.dilate(grasp_mor_segment, grasp_kernel)
        else:
            grasp_mor_dilate = cv2.erode(grasp_mor_segment, grasp_kernel)
        _, contours, hierarchy = cv2.findContours(grasp_mor_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10:
                rect = cv2.minAreaRect(contour)
                center_x, center_y = rect[0]
                cv2.circle(grasp_mor_gradient, (int(round(center_x)), int(round(center_y))), 2, (255, 0, 0), -1)   # 2
        # visualize
        cv2.imshow('grasp-mask', grasp_mor_gradient)
        cv2.waitKey(100)

        # push_mask
        push_mor_erode = None
        push_binary_segment = stuff_count.copy()
        push_kernel = np.ones((7, 7), np.uint8) 
        push_delete_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        push_mor_segment = cv2.morphologyEx(push_binary_segment, cv2.MORPH_OPEN, push_delete_small)

        push_dilate_kernel = np.ones((15, 15), np.uint8)
        push_dilate_1 = cv2.dilate(push_mor_segment, push_kernel)
        push_dilate_2 = cv2.dilate(push_mor_segment, push_dilate_kernel)
        push_mor_gradient = push_dilate_2 - push_dilate_1
        # visualize
        cv2.imshow('push-mask', push_mor_gradient)
        cv2.waitKey(100)

        # Restart a new epoch if the following conditions are met
        if Fail or (epoch_count % 25 == 0) or np.sum(black_count) < empty_threshold or (is_sim and no_change_count[0] + no_change_count[1] > 15):
            if is_sim:
                # the epoch number more than 25(fail)
                if (epoch_count % 25 == 0):
                    print('This epoch is Over, Count == 25 !!')
                    time.sleep(6)
                # empty(if prev_num_in_workspace == 0, success)
                if np.sum(black_count) < empty_threshold:
                    print('Not enough objects in view (value: %d but the threshold: %d)! Repositioning objects.' % (np.sum(black_count), empty_threshold))
                    time.sleep(6)
                    if prev_num_in_workspace == 0:
                        Win = True
                # no_change >= 15(fail)
                if (is_sim and no_change_count[0] + no_change_count[1] > 15):
                    print('More Than 15 times No Change Detected')
                    time.sleep(6)
                # outside
                if Fail:
                    print('some objects are pushed out of the workspace!!')
                    time.sleep(6)
                    Fail = False
                    trainer.pushout_log.append([trainer.iteration])
                    logger.write_to_log('pushout', trainer.pushout_log)

                # new epoch
                robot.del_rechoice()
                judge_reset = True
                # record the epoch number
                trainer.success_log.append([trainer.iteration, epoch_count])
                logger.write_to_log('success', trainer.success_log)
                epoch_count = 1
            else:
                # the epoch number more than 25(fail)
                if (epoch_count % 25 == 0):
                    print('This epoch is Over, Count == 25 !! Please reset objects!!!')
                    time.sleep(10)
                # empty
                if np.sum(black_count) < empty_threshold:
                    print('Not enough objects in view (value: %d but the threshold: %d)! Repositioning objects.' % (np.sum(black_count), empty_threshold))
                    time.sleep(10)

                judge_reset = True
                epoch_count = 1

            # re-load original weights if in test mode
            if is_testing:
                trainer.push_model.load_state_dict(torch.load(push_snapshot_file))
                trainer.grasp_model.load_state_dict(torch.load(grasp_snapshot_file))

            no_change_count = [0, 0]
            trainer.clearance_log.append([trainer.iteration]) 
            logger.write_to_log('clearance', trainer.clearance_log)
            if is_testing and len(trainer.clearance_log) >= max_test_trials:
                exit_called = True # Exit after training thread (backprop and saving labels)
            continue
        else:
            print('Current objects Pixels in view (value: %d) more than threshold (value: %d).' % (np.sum(black_count), empty_threshold))

        if not exit_called:
            if is_testing and epoch_count % 3 == 0:  # re-load original weights in 3 step (before test run)
                trainer.push_model.load_state_dict(torch.load(push_snapshot_file))
                trainer.grasp_model.load_state_dict(torch.load(grasp_snapshot_file))

            # forward
            push_predictions, grasp_predictions, _ = trainer.forward(color_heightmap, valid_depth_heightmap, is_volatile=True)
            grasp_predictions = grasp_predictions + 0.1
            push_predictions = push_predictions + 0.1
            # action_mask(push)
            push_action_mask = np.array(push_mor_gradient.copy()/255).reshape(1,224,224)
            push_predictions = np.multiply(push_predictions, push_action_mask)
            # action_mask(grasp)
            grasp_action_mask = np.array(grasp_mor_gradient.copy()/255).reshape(1,224,224)
            grasp_predictions = np.multiply(grasp_predictions, grasp_action_mask)

            # Execute best primitive action on robot in another thread
            nonlocal_variables['executing_action'] = True

        # Run training iteration in current thread (training thread)
        if 'prev_color_img' in locals():
            # Detect changes
            depth_diff = abs(depth_heightmap - prev_depth_heightmap)
            if is_sim:
                change_threshold = 580
            else:
                change_threshold = 1000

            depth_diff[np.isnan(depth_diff)] = 0
            depth_diff[depth_diff > 0.3] = 0
            depth_diff[depth_diff < 0.02] = 0
            depth_diff[depth_diff > 0] = 1
            change_value = np.sum(depth_diff)
            change_detected = (change_value > change_threshold or prev_grasp_success) and not judge_reset
            print('Change detected: %r (value: %d , whether because reset: %s)' % (change_detected, change_value, judge_reset))

            # get reward
            label_value, prev_reward_value, Fail, Win = trainer.get_label_value(prev_primitive_action, prev_grasp_success, change_detected, color_heightmap, valid_depth_heightmap, prev_reward_depth_heightmap, reward_depth_heightmap, inside, judge_reset, Win)

            if change_detected:
                if prev_primitive_action == 'push':
                    no_change_count[0] = 0
                elif prev_primitive_action == 'grasp':
                    no_change_count[1] = 0
            else:
                judge_reset = False
                if prev_primitive_action == 'push':
                    no_change_count[0] += 1
                elif prev_primitive_action == 'grasp':
                    no_change_count[1] += 1

            trainer.label_value_log.append([label_value])
            logger.write_to_log('label-value', trainer.label_value_log)
            trainer.reward_value_log.append([prev_reward_value])
            logger.write_to_log('reward-value', trainer.reward_value_log)

            # Backpropagate
            trainer.backprop(prev_color_heightmap, prev_valid_depth_heightmap, prev_primitive_action, prev_best_pix_ind, label_value)

            # explore rate
            if not is_testing:
                explore_prob = max(0.4 * np.power(0.9995, trainer.iteration), 0.1) if explore_rate_decay else 0.5

            # Do sampling for experience replay
            if experience_replay and not is_testing:
                sample_primitive_action = prev_primitive_action
                if sample_primitive_action == 'push':
                    sample_primitive_action_id = 0
                    sample_reward_value = 0 if prev_reward_value >= 0.5 else 0.5
                elif sample_primitive_action == 'grasp':
                    sample_primitive_action_id = 1
                    sample_reward_value = 0 if prev_reward_value >= 0.5 else 0.5

                if sample_reward_value == 0:
                    sample_ind = np.argwhere(np.logical_and(np.asarray(trainer.reward_value_log)[0:trainer.iteration,0] < sample_reward_value + 0.5, np.asarray(trainer.executed_action_log)[0:trainer.iteration,0] == sample_primitive_action_id))
                elif sample_reward_value == 0.5:
                    sample_ind = np.argwhere(np.logical_and(np.asarray(trainer.reward_value_log)[0:trainer.iteration,0] >= sample_reward_value, np.asarray(trainer.executed_action_log)[0:trainer.iteration,0] == sample_primitive_action_id))
                
                if sample_ind.size > 0:
                    # Find sample with highest surprise value
                    sample_surprise_values = np.abs(np.asarray(trainer.predicted_value_log)[sample_ind[:,0]] - np.asarray(trainer.label_value_log)[sample_ind[:,0]])
                    sorted_surprise_ind = np.argsort(sample_surprise_values[:,0]) 
                    sorted_sample_ind = sample_ind[sorted_surprise_ind,0]

                    pow_law_exp = 2
                    rand_sample_ind = int(np.round(np.random.power(pow_law_exp, 1)*(sample_ind.size-1)))
                    sample_iteration = sorted_sample_ind[rand_sample_ind]
                    print('Experience replay: iteration %d (surprise value: %f)' % (sample_iteration, sample_surprise_values[sorted_surprise_ind[rand_sample_ind]]))

                    # Load sample RGB-D heightmap
                    sample_color_heightmap = cv2.imread(os.path.join(logger.color_heightmaps_directory, '%06d.0.color.png' % (sample_iteration)))
                    sample_color_heightmap = cv2.cvtColor(sample_color_heightmap, cv2.COLOR_BGR2RGB)
                    sample_depth_heightmap = cv2.imread(os.path.join(logger.depth_heightmaps_directory, '%06d.0.depth.png' % (sample_iteration)), -1)
                    sample_depth_heightmap = sample_depth_heightmap.astype(np.float32)/100000

                    # Compute forward pass with sample
                    sample_push_predictions, sample_grasp_predictions, _ = trainer.forward(sample_color_heightmap, sample_depth_heightmap, is_volatile=True)
                    # Load next sample RGB-D heightmap
                    next_sample_color_heightmap = cv2.imread(os.path.join(logger.color_heightmaps_directory, '%06d.0.color.png' % (sample_iteration+1)))
                    next_sample_color_heightmap = cv2.cvtColor(next_sample_color_heightmap, cv2.COLOR_BGR2RGB)
                    next_sample_depth_heightmap = cv2.imread(os.path.join(logger.depth_heightmaps_directory, '%06d.0.depth.png' % (sample_iteration+1)), -1)
                    next_sample_depth_heightmap = next_sample_depth_heightmap.astype(np.float32)/100000

                    # Get labels for sample and backpropagate
                    sample_best_pix_ind = (np.asarray(trainer.executed_action_log)[sample_iteration, 1:4]).astype(int)
                    trainer.backprop(sample_color_heightmap, sample_depth_heightmap, sample_primitive_action, sample_best_pix_ind, trainer.label_value_log[sample_iteration])

                    if sample_primitive_action == 'push':
                        trainer.predicted_value_log[sample_iteration] = [np.max(sample_push_predictions)]
                    elif sample_primitive_action == 'grasp':
                        trainer.predicted_value_log[sample_iteration] = [np.max(sample_grasp_predictions)]
                else:
                    print('Not enough prior training samples. Skipping experience replay.')

            if not is_testing:
                # save param
                logger.save_multi_backup_model([trainer.push_model, trainer.grasp_model])
                if trainer.iteration % 1000 == 0:
                    torch.cuda.empty_cache()
                    logger.save_multi_model(trainer.iteration, [trainer.push_model,trainer.grasp_model])
                    if trainer.use_cuda:
                        trainer.push_model = trainer.push_model.cuda()
                        trainer.grasp_model = trainer.grasp_model.cuda()

        # Sync both action thread and training thread
        while nonlocal_variables['executing_action']:
            time.sleep(0.1)

        if exit_called: 
            break

        prev_color_img = color_img.copy()
        prev_depth_img = depth_img.copy()
        prev_color_heightmap = color_heightmap.copy()
        prev_depth_heightmap = depth_heightmap.copy()
        prev_valid_depth_heightmap = valid_depth_heightmap.copy()
        prev_push_success = nonlocal_variables['push_success']
        prev_grasp_success = nonlocal_variables['grasp_success']
        prev_primitive_action = nonlocal_variables['primitive_action']
        prev_push_predictions = push_predictions.copy()
        prev_grasp_predictions = grasp_predictions.copy()
        prev_reward_depth_heightmap = reward_depth_heightmap.copy()
        prev_best_pix_ind = nonlocal_variables['best_pix_ind']
        prev_num_in_workspace = nonlocal_variables['num_in_workspace']
        trainer.iteration += 1
        epoch_count += 1
        
        # wait for the operation have finished then got the next color and depth image
        if not is_sim:
            time.sleep(1)
        iteration_time_1 = time.time()
        print('Time elapsed: %f' % (iteration_time_1-iteration_time_0))


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train robotic agents to learn how to plan complementary pushing and grasping actions for manipulation with deep reinforcement learning in PyTorch.')

    # --------------- Setup options ---------------
    parser.add_argument('--is_sim', dest='is_sim', action='store_true', default=True,                                    help='run in simulation?')
    parser.add_argument('--random_seed', dest='random_seed', type=int, action='store', default=123456,                      help='random seed for simulation and neural net initialization')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,                                    help='force code to run in CPU mode')

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    # ------------- Algorithm options -------------
    parser.add_argument('--push_rewards', dest='push_rewards', action='store_true', default=True,                        help='use immediate rewards (from change detection) for pushing?')
    parser.add_argument('--future_reward_discount', dest='future_reward_discount', type=float, action='store', default=0.5)
    parser.add_argument('--experience_replay', dest='experience_replay', action='store_true', default=True,              help='use prioritized experience replay?')
    parser.add_argument('--heuristic_bootstrap', dest='heuristic_bootstrap', action='store_true', default=False,          help='use handcrafted grasping algorithm when grasping fails too many times in a row during training?')
    parser.add_argument('--explore_rate_decay', dest='explore_rate_decay', action='store_true', default=True)

    # -------------- Testing options --------------
    parser.add_argument('--is_testing', dest='is_testing', action='store_true', default=False)
    parser.add_argument('--max_test_trials', dest='max_test_trials', type=int, action='store', default=25,                help='maximum number of test runs per case/scenario')

    # ------ Pre-loading and logging options ------
    parser.add_argument('--continue_logging', dest='continue_logging', action='store_true', default=False,                help='continue logging from previous session?')
    parser.add_argument('--logging_directory', dest='logging_directory', action='store')
    parser.add_argument('--save_visualizations', dest='save_visualizations', action='store_true', default=True,          help='save visualizations of FCN predictions?')

    parser.add_argument('--load_snapshot', dest='load_snapshot', action='store_true', default=True,                      help='load pre-trained snapshot of model?')
    parser.add_argument('--push_snapshot_file', dest='push_snapshot_file', action='store', default='/home/user/myur_ws/src/drl_push_grasp/logs/push_pre_train/models/snapshot-backup.push.pth')  # 
    parser.add_argument('--grasp_snapshot_file', dest='grasp_snapshot_file', action='store', default='/home/user/myur_ws/src/drl_push_grasp/logs/grasp_pre_train/models/snapshot-backup.grasp.pth')   # 

    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)
