#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
##########################################################
calibrate_myrobot.py: Extrinsics Calibration(Eye-to-hand)
##########################################################
'''
import numpy as np 
import rospy
import actionlib
from control_msgs.msg import *
from trajectory_msgs.msg import *
from sensor_msgs.msg import JointState
from tf import TransformListener
from math import pi 
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import sys, tf
from gazebo_msgs.srv import *
from geometry_msgs.msg import *
from std_srvs.srv import Empty
from std_msgs.msg import UInt16
from copy import deepcopy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
import random
from matplotlib import pyplot as plt

import moveit_commander
from moveit_commander import MoveGroupCommander
from geometry_msgs.msg import Pose
import roslib
import rospy
import actionlib

from dh_hand_driver.msg import ActuateHandAction, ActuateHandGoal

import time
from real.camera import Camera
from scipy import optimize  


# --------------- Setup options ----------------
workspace_limits = np.asarray([[-0.48, 0.17], [0.35, 0.80], [0.31, 0.41]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
calib_grid_step = 0.05

checkerboard_offset_from_tool = [0,+0.06,0.025]   # the offset of gripper and calibration-target
tool_orientation = [-np.pi/2,0,0]


# ----------------Camera Start-------------------
camera = Camera()
cam_intrinsics = camera.intrinsics


#-----------------ROS Moveit initial-------------
rospy.init_node('test', anonymous=True)
moveit_commander.roscpp_initialize(sys.argv)
cartesian = rospy.get_param('~cartesian', True)
arm = moveit_commander.MoveGroupCommander('manipulator')
end_effector_link = arm.get_end_effector_link()

reference_frame = '/world'
arm.set_pose_reference_frame(reference_frame)
arm.allow_replanning(True)
arm.set_goal_position_tolerance(0.001)
arm.set_goal_orientation_tolerance(0.01)
arm.set_max_acceleration_scaling_factor(0.1)
arm.set_max_velocity_scaling_factor(0.2)

#---------------Gripper connection-------------
client = actionlib.SimpleActionClient('actuate_hand', ActuateHandAction)
client.wait_for_server()


def close_gripper():
    goal = ActuateHandGoal()
    goal.MotorID = 1    #motoID
    goal.force = 20   #Force
    goal.position = 0    #Position
    client.send_goal(goal)
    client.wait_for_result(rospy.Duration.from_sec(4.0))


def open_gripper():
    goal = ActuateHandGoal()
    goal.MotorID = 1    #motoID
    goal.force = 20   #Force
    goal.position = 100    #Position
    client.send_goal(goal)
    client.wait_for_result(rospy.Duration.from_sec(4.0))


def move_to_initial():
    joint_positions = [np.pi/2, -np.pi/2, np.pi/2, 0, np.pi/2, np.pi/2]
    arm.set_joint_value_target(joint_positions)
    arm.go()
    rospy.sleep(1)

 
def calibrate_move(movePos):
    start_pose = arm.get_current_pose(end_effector_link).pose
    print(start_pose)

    waypoints = []

    # Set KeyPoint data and add to the KeyPoint list
    target_pose1 = PoseStamped()
    target_pose1.header.frame_id = reference_frame
    target_pose1.header.stamp = rospy.Time.now()     
    target_pose1.pose.position.x = movePos[0]
    target_pose1.pose.position.y = movePos[1]
    target_pose1.pose.position.z = movePos[2]
    target_pose1.pose.orientation.x = start_pose.orientation.x
    target_pose1.pose.orientation.y = start_pose.orientation.y
    target_pose1.pose.orientation.z = start_pose.orientation.z
    target_pose1.pose.orientation.w = start_pose.orientation.w
    waypoints.append(target_pose1.pose)

    fraction = 0.0
    maxtries = 100
    attempts = 0

    arm.set_start_state_to_current_state()

    # Planning the Cartesian coordinates and pass the all KeyPoint.
    while fraction < 1.0 and attempts < maxtries:
        (plan, fraction) = arm.compute_cartesian_path (
                                waypoints,
                                0.001,
                                0.0,
                                True)
        
        attempts += 1
                
    # if success, move the ur robotic arm
    if fraction == 1.0:
        arm.execute(plan)
    # fail
    else:
        rospy.loginfo("Path planning failed")


# Construct 3D calibration grid across workspace
gridspace_x = np.linspace(workspace_limits[0][0], workspace_limits[0][1], 13) # 8
gridspace_y = np.linspace(workspace_limits[1][0], workspace_limits[1][1], 9)  # 9
gridspace_z = np.linspace(workspace_limits[2][0], workspace_limits[2][1], 3)  # 3
print(gridspace_x)
print(gridspace_y)
print(gridspace_z)

calib_grid_x, calib_grid_y, calib_grid_z = np.meshgrid(gridspace_x, gridspace_y, gridspace_z)
print(calib_grid_y.shape)

num_calib_grid_pts = calib_grid_x.shape[0]*calib_grid_x.shape[1]*calib_grid_x.shape[2]   #9*8*3 = 216
calib_grid_x.shape = (num_calib_grid_pts,1)
calib_grid_y.shape = (num_calib_grid_pts,1)
calib_grid_z.shape = (num_calib_grid_pts,1)

calib_grid_pts = np.concatenate((calib_grid_x, calib_grid_y, calib_grid_z), axis=1)
print(calib_grid_pts)

measured_pts = []
observed_pts = []
observed_pix = []

# Move robot to home pose
move_to_initial()
close_gripper()
time.sleep(3)
open_gripper()

# Move robot to each calibration point in workspace
print('Collecting data...')

for calib_pt_idx in range(num_calib_grid_pts):
    tool_position = calib_grid_pts[calib_pt_idx,:]
    calibrate_move(tool_position)
    time.sleep(2)
    
    # Find checkerboard center
    checkerboard_size = (3, 3)
    refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    camera_color_img, camera_depth_img = camera.get_data()
    bgr_color_data = cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)
    gray_data = cv2.cvtColor(bgr_color_data, cv2.COLOR_RGB2GRAY)
    checkerboard_found, corners = cv2.findChessboardCorners(gray_data, checkerboard_size, None, cv2.CALIB_CB_ADAPTIVE_THRESH)
    if checkerboard_found:
        corners_refined = cv2.cornerSubPix(gray_data, corners, (3,3), (-1,-1), refine_criteria)

        # Get observed checkerboard center 3D point in camera space
        checkerboard_pix = np.round(corners_refined[4,0,:]).astype(int)
        checkerboard_z = camera_depth_img[checkerboard_pix[1]][checkerboard_pix[0]]
        checkerboard_x = np.multiply(checkerboard_pix[0]-cam_intrinsics[0][2],checkerboard_z/cam_intrinsics[0][0])
        checkerboard_y = np.multiply(checkerboard_pix[1]-cam_intrinsics[1][2],checkerboard_z/cam_intrinsics[1][1])
        if checkerboard_z == 0:
            continue

        # Save calibration point and observed checkerboard center
        observed_pts.append([checkerboard_x, checkerboard_y, checkerboard_z])
        tool_position = tool_position + checkerboard_offset_from_tool

        measured_pts.append(tool_position)
        observed_pix.append(checkerboard_pix)

        # Draw and display the corners
        vis = cv2.drawChessboardCorners(bgr_color_data, (1,1), corners_refined[4,:,:], checkerboard_found)
        cv2.imwrite('/home/user/myur_ws/src/drl_push/scripts/camera_instrinsic/%06d.png' % len(measured_pts), vis)
        cv2.imshow('Calibration',vis)
        cv2.waitKey(10)

# Move robot to home pose
move_to_initial()


#--------------------------caculate the extrinsics----------------------------
measured_pts = np.asarray(measured_pts)  # world coordinates
observed_pts = np.asarray(observed_pts)  # camera coordinates
observed_pix = np.asarray(observed_pix)  # pixel coordinates
world2camera = np.eye(4)


# Estimate rigid transform with SVD (from Nghia Ho)
def get_rigid_transform(A, B):       # A is the world coordinates，B is the camera coordinates
    assert len(A) == len(B)
    N = A.shape[0]; # Total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - np.tile(centroid_A, (N, 1)) # Centre the points
    BB = B - np.tile(centroid_B, (N, 1))
    H = np.dot(np.transpose(AA), BB) # Dot is matrix multiplication for array
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0: # Special reflection case
       Vt[2,:] *= -1
       R = np.dot(Vt.T, U.T)
    t = np.dot(-R, centroid_A.T) + centroid_B.T
    return R, t


def get_rigid_transform_error(z_scale):
    global measured_pts, observed_pts, observed_pix, world2camera, camera

    # Apply z offset and compute new observed points using camera intrinsics
    observed_z = observed_pts[:,2:] * z_scale
    observed_x = np.multiply(observed_pix[:,[0]]-cam_intrinsics[0][2],observed_z/cam_intrinsics[0][0])
    observed_y = np.multiply(observed_pix[:,[1]]-cam_intrinsics[1][2],observed_z/cam_intrinsics[1][1])
    new_observed_pts = np.concatenate((observed_x, observed_y, observed_z), axis=1)

    # Estimate rigid transform between measured points and new observed points
    R, t = get_rigid_transform(np.asarray(measured_pts), np.asarray(new_observed_pts))
    t.shape = (3,1)
    world2camera = np.concatenate((np.concatenate((R, t), axis=1),np.array([[0, 0, 0, 1]])), axis=0)

    # Compute rigid transform error
    registered_pts = np.dot(R, np.transpose(measured_pts)) + np.tile(t,(1,measured_pts.shape[0]))
    error = np.transpose(registered_pts) - new_observed_pts
    error = np.sum(np.multiply(error, error))
    rmse = np.sqrt(error/measured_pts.shape[0]);
    return rmse

# Optimize z scale w.r.t. rigid transform error
print('Calibrating...')
z_scale_init = 1
optim_result = optimize.minimize(get_rigid_transform_error, np.asarray(z_scale_init), method='Nelder-Mead')
camera_depth_offset = optim_result.x

# Save camera optimized offset and camera pose
print('Saving...')
np.savetxt('/home/user/myur_ws/src/drl_push_grasp/scripts/real/camera_depth_scale.txt', camera_depth_offset, delimiter=' ')
get_rigid_transform_error(camera_depth_offset)
camera_pose = np.linalg.inv(world2camera)
np.savetxt('/home/user/myur_ws/src/drl_push_grasp/scripts/real/camera_pose.txt', camera_pose, delimiter=' ')
print('Done.')

