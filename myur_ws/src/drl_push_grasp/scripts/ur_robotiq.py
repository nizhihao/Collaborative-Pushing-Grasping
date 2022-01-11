#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
########################################################################################
ur_robotiq.py: the environment, state obtainment, action execution，robot controll, etc.
########################################################################################
'''
import numpy as np 
import rospy
import actionlib
import roslib
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
import os
import time
import random
import utils

import moveit_commander
from moveit_commander import MoveGroupCommander
from geometry_msgs.msg import Pose

from real.camera import Camera

from dh_hand_driver.msg import ActuateHandAction, ActuateHandGoal
from dh_hand_driver.srv import hand_state

# random seed
np.random.seed(1234)

# push and grasp
USE_OBJECTS = ['cappuccino', 'chewinggum', 'cleaner', 'coke_can', 'cube', 'cup', 'eraser', 'glue',
               'rice', 'shampoo', 'sticky_notes', 'sugar', 'teagreen', 'teayellow', 'thuna', 'salt', 'sweetener']

# initial scene for training
INITIAL_POS = ['3_1', '3_2', '3_3', '4_1', '4_2', '4_3', '4_4', '5_1', '5_2', '5_3', '6_1', '6_2', '6_3', '6_4']

# initial scene for testing
INITIAL_TEST_POS = ['3_1', '3_2', '3_3', '4_1', '4_2', '4_3', '4_4', '5_1', '5_2', '5_3', '6_1', '6_2', '6_3', '6_4', '7_1', '7_2', '8_1', '8_2']

JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

#joint motion duration
DURATION = 0.5 
GOAL = [0.75, 0, 0.639239, 0, 0, 0]
#initial joint angle
INIT = [0, -pi/4, pi/3, -2*pi/3, -pi/2, 0]

Box = [pi/6, -pi/2, pi/4, -pi/2, -pi/2, 0]

class Ur10(object):

    def __init__(self, is_sim, is_testing, workspace_limits, init_joints=INIT, goal_pose=GOAL,duration=DURATION):
        self.is_sim = is_sim
        self.workspace_limits = workspace_limits
        self.is_testing = is_testing

        # create the node
        rospy.init_node('ur10_env', anonymous=True)
        parameters = rospy.get_param(None)
        index = str(parameters).find('prefix')
        if (index > 0):
            prefix = str(parameters)[index+len("prefix': '"):(index+len("prefix': '")+str(parameters)[index+len("prefix': '"):-1].find("'"))]
            for i, name in enumerate(JOINT_NAMES):
                JOINT_NAMES[i] = prefix + name

        # Gazebo
        if self.is_sim:
            # Define the ROS communication of Sevice or Action or Topic 
            self.spawn = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel,persistent=True)
            self.delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel,persistent=True)
            self.set_oject_position_service = rospy.ServiceProxy('gazebo/set_model_state', SetModelState,persistent=True)
            self.get_oject_position_service = rospy.ServiceProxy('gazebo/get_model_state', GetModelState,persistent=True)
            self.client = actionlib.SimpleActionClient('/arm_controller/follow_joint_trajectory',
                                                                    FollowJointTrajectoryAction)
            self.client_gripper = actionlib.SimpleActionClient('/gripper_controller/follow_joint_trajectory',
                                                                    FollowJointTrajectoryAction)
            self.client.wait_for_server()
            print('connect to server')

            self.client_gripper.wait_for_server()
            print('connect to gripper_server')

            # moveit_command init
            moveit_commander.roscpp_initialize(sys.argv)
            cartesian = rospy.get_param('~cartesian', True)
            self.arm = moveit_commander.MoveGroupCommander('manipulator')
            self.end_effector_link = self.arm.get_end_effector_link()   # wrist3_link
            
            self.reference_frame = 'world'
            self.arm.set_pose_reference_frame(self.reference_frame)
            self.arm.allow_replanning(True)
            self.arm.set_goal_position_tolerance(0.001)
            self.arm.set_goal_orientation_tolerance(0.01)
            self.arm.set_max_acceleration_scaling_factor(0.24)
            self.arm.set_max_velocity_scaling_factor(0.48) 
                                                    
            self.tf = TransformListener()
            self.duration = duration
            self.gripper(action=[0])     # note: 1.0 means to close the gripper，0.0 means to open the gripper
            self.gripper_closed = False
            self.num = 0
            self.cube_name = []

            # get the intrinsics and extrinsic matrixs
            self.setup_sim_camera()
            # spwan all object to the Gazebo
            self.input_all_object()
        # Real-World
        else:
            self.get_gripper_pose = rospy.ServiceProxy('/hand_joint_state', hand_state)
            self.client = actionlib.SimpleActionClient('/follow_joint_trajectory',FollowJointTrajectoryAction)
            self.client_gripper = actionlib.SimpleActionClient('/actuate_hand', ActuateHandAction)
            self.client.wait_for_server()
            print('connect to server')

            self.client_gripper.wait_for_server()
            print('connect to gripper_server')

            # moveit_command init
            moveit_commander.roscpp_initialize(sys.argv)
            cartesian = rospy.get_param('~cartesian', True)
            self.arm = moveit_commander.MoveGroupCommander('manipulator')
            self.end_effector_link = self.arm.get_end_effector_link()
            
            self.reference_frame = 'world'
            self.arm.set_pose_reference_frame(self.reference_frame)
            self.arm.allow_replanning(True)
            self.arm.set_goal_position_tolerance(0.001)
            self.arm.set_goal_orientation_tolerance(0.01)
            self.arm.set_max_acceleration_scaling_factor(0.24)
            self.arm.set_max_velocity_scaling_factor(0.38)

            self.tf = TransformListener()
            self.duration = duration
            self.real_gripper(0)    #   0 - close, 100 - open
            self.gripper_closed = True

            self.camera = Camera()
            self.setup_real_camera()
            self.real_reset()


    def get_camera_data(self):
        # get color img and depth img
        if self.is_sim:
            # Get color image from simulation
            rospy.sleep(1)
            raw_img = rospy.wait_for_message('/head_mount_kinect2/rgb/image_raw', Image)
            raw_image = CvBridge().imgmsg_to_cv2(raw_img,"rgb8")

            color_img = np.asarray(raw_image)
            color_img = color_img.astype(np.float)/255
            color_img[color_img < 0] += 1
            color_img *= 255
            color_img = color_img.astype(np.uint8)

            # Get depth image from simulation
            depth_img = rospy.wait_for_message('/head_mount_kinect2/depth/image_raw', Image)
            rospy.sleep(0.01)
            depth_image = CvBridge().imgmsg_to_cv2(depth_img)   # (480, 640)
            depth_img = np.asarray(depth_image)
        else:
            color_img, depth_img = self.camera.get_data()

        return color_img, depth_img


    def input_all_object(self):
        self.initial_arm()
        self.object_spwan()

        if self.gripper_closed:
            self.gripper(action=[0])  #1.0 - close，0.0 - open
            self.gripper_closed = False  

        if not self.is_testing:
            self.random_choice_env()
        else:
            self.choice_test_env()


    def initial_arm(self):
        # controll the ur to the init state, You can set your init state by moveit assistant setup
        self.arm.set_named_target('init')
        self.arm.go()
        rospy.sleep(0.1)


    def object_spwan(self):
        # initial all objects, and spwan them to the Gazebo 
        rospy.wait_for_service("gazebo/spawn_sdf_model",timeout=5)

        for i in USE_OBJECTS:
            rand_x, rand_y = np.random.uniform(-0.5,0.5), np.random.uniform(-1,1)
            goal = np.array([-1.5,0,0.1,0,0,0.7853981633])
            goal[0] = goal[0] + rand_x
            goal[1] = goal[1] + rand_y

            orient = Quaternion(*tf.transformations.quaternion_from_euler(goal[3],goal[4],goal[5]))
            origin_pose = Pose(Point(goal[0],goal[1],goal[2]), orient)

            filename = '/home/user/myur_ws/src/ur_robotiq/ur_robotiq_gazebo/meshes/%s/model.sdf'%(i)
            with open(filename,"r") as f:
                reel_xml = f.read()

            pose = deepcopy(origin_pose)
            self.spawn(i, reel_xml, "", pose, "world")
            time.sleep(1)


    def gripper(self, action):
        # controll the gripper by Action communication in the Gazebo
        g = FollowJointTrajectoryGoal()
        g.trajectory = JointTrajectory()
        g.trajectory.joint_names = ['gripper_finger1_joint']

        try:
            joint_states = rospy.wait_for_message("joint_states", JointState)
            joints_pos = joint_states.position
            g.trajectory.points = [
                JointTrajectoryPoint(positions=joints_pos, velocities=[0]*1, time_from_start=rospy.Duration(0.0)),
                JointTrajectoryPoint(positions=action, velocities=[0]*1, time_from_start=rospy.Duration(DURATION)),]
            self.client_gripper.send_goal(g)
            self.client_gripper.wait_for_result()
        except KeyboardInterrupt:
            self.client_gripper.cancel_goal()
            raise


    def real_gripper(self, position):
        # controll the dh gripper in real world
        goal = ActuateHandGoal()
        goal.MotorID = 1             # motoID
        goal.force = 30             # Force   30
        goal.position = position    # Position between 0-100
        # Fill in the goal here
        self.client_gripper.send_goal(goal)
        self.client_gripper.wait_for_result(rospy.Duration.from_sec(5.0))


    def random_choice_env(self):
        # train, read the objects' postion and orientation from the txt
        initial_index = np.random.randint(0, len(INITIAL_POS))
        num_ind = INITIAL_POS[initial_index]
        a = num_ind.split('_')
        base_path = '/home/user/myur_ws/src/ur_robotiq/ur_robotiq_gazebo/test_obj(grasp_only)'
        target_path = os.path.join(base_path, a[0], a[1]) + '/%s.txt'%(a[1])
        print('Cur_Initial_env: {}/{}.txt'.format(a[0], a[1]))

        with open(target_path, 'r') as fd:
            obj_pos = fd.readlines()

        target_obj = []
        for i in obj_pos:
            i = i.rstrip()
            print(i)
            name, pos = i.split()
            self.cube_name.append(name)
            pos = pos.split(',')
            target_obj.append([name, [float(j) for j in pos]])

        self.move2desk(target_obj)


    def choice_test_env(self):
        # evaluation, read the objects' postion and orientation from the txt
        initial_index = np.random.randint(0, len(INITIAL_TEST_POS))
        num_ind = INITIAL_TEST_POS[initial_index]
        a = num_ind.split('_')
        base_path = '/home/user/myur_ws/src/ur_robotiq/ur_robotiq_gazebo/test_obj(grasp_only)'
        target_path = os.path.join(base_path, a[0], a[1]) + '/%s.txt' % (a[1])
        print('Cur_Initial_test_env: {}/{}.txt'.format(a[0], a[1]))

        with open(target_path, 'r') as fd:
            obj_pos = fd.readlines()

        target_obj = []
        for i in obj_pos:
            i = i.rstrip()
            print(i)
            name, pos = i.split()
            self.cube_name.append(name)
            pos = pos.split(',')
            # test with pos random
            pos[5] = float(pos[5]) + (np.random.sample() - 0.5) * 2 * 3.14
            target_obj.append([name, [float(j) for j in pos]])

        self.move2desk(target_obj)


    def move2desk(self, name_pos_list):
        # move some object to desk, initial state
        for obj in name_pos_list:
            # obj[0]: obj_name, obj[1]: obj_pos and obj_orientation
            orient = Quaternion(*tf.transformations.quaternion_from_euler(obj[1][3], obj[1][4], obj[1][5])) 
            objstate = SetModelStateRequest()
            objstate.model_state.model_name = obj[0]
            objstate.model_state.pose.position.x = obj[1][0]
            objstate.model_state.pose.position.y = obj[1][1]
            objstate.model_state.pose.position.z = obj[1][2]
            objstate.model_state.pose.orientation = orient
            objstate.model_state.reference_frame = "world"
            self.set_oject_position_service(objstate)
            time.sleep(0.5)


    def real_reset(self):
        # reset the state of the UR arm
        joint_positions = [np.pi/2, -105*np.pi/180, 80*np.pi/180, -65*np.pi/180, -np.pi/2, np.pi/2]
        self.arm.set_joint_value_target(joint_positions)
        self.arm.go()

        if not self.gripper_closed:
            self.real_gripper(0)
            self.gripper_closed = True

        rospy.sleep(0.1)


    def setup_real_camera(self):
        self.cam_pose = np.loadtxt('/home/user/myur_ws/src/drl_push/scripts/real/camera_pose.txt', delimiter=' ')
        self.cam_depth_scale = np.loadtxt('/home/user/myur_ws/src/drl_push/scripts/real/camera_depth_scale.txt', delimiter=' ')
        self.cam_intrinsics = self.camera.intrinsics


    def setup_sim_camera(self):
        # you can get the position and orientation by the command line: rosrun tf tf_echo /world /YOU_CAMERA_LINK  or  rosrun tf tf_echo /YOU_CAMERA_LINK /world 
        cam_position = [0.750, -0.508, 1.493]
        cam_orientation = [2.617994, 0.000, -3.1415926]
        
        cam_trans = np.eye(4,4)
        cam_trans[0:3,3] = np.asarray(cam_position)
        
        cam_orientation = [cam_orientation[0], cam_orientation[1], cam_orientation[2]]
        cam_rotm = np.eye(4,4)
        cam_rotm[0:3,0:3] = np.linalg.inv(utils.euler2rotm(cam_orientation))
        self.cam_pose = np.dot(cam_trans, cam_rotm)

        # you can get the cam_intrinsics by the command line: rostopic echo /camera_info
        self.cam_intrinsics = np.asarray([[589.3664541825391, 0.0, 320.5], [0.0, 589.3664541825391, 240.5], [0, 0, 1]])
        self.cam_depth_scale = 1


    def del_rechoice(self):
        # rechoice one init env, run a new epoch
        for obj_name in self.cube_name:
            self.out_obj_pos(obj_name)
        
        self.cube_name = []
        if not self.is_testing:
            self.random_choice_env()
        else:
            self.choice_test_env()


    def out_obj_pos(self,obj_name):
        # output the obj that in the workspace
        rand_x, rand_y = np.random.uniform(-0.5,0.5), np.random.uniform(-1,1)
        orient = Quaternion(*tf.transformations.quaternion_from_euler(0,0,0.7853981633)) 
        
        objstate = SetModelStateRequest()
        objstate.model_state.model_name = obj_name
        objstate.model_state.pose.position.x = -1.5 + rand_x
        objstate.model_state.pose.position.y = rand_y
        objstate.model_state.pose.position.z = 0.1
        objstate.model_state.pose.orientation = orient
        objstate.model_state.reference_frame = "world"
        
        self.set_oject_position_service(objstate)


    def get_env_obj_pos(self):
        # judge the objects in or not in the workspace
        if self.cube_name:
            Get_Pos = GetModelStateRequest()
            for i in self.cube_name:
                Get_Pos.model_name = i
                obj_x = self.get_oject_position_service(Get_Pos).pose.position.x
                obj_y = self.get_oject_position_service(Get_Pos).pose.position.y
                # workspace [0.414, 1.086], [-0.336, 0.336], [0.60, 1]
                if obj_x < 0.358 or obj_x > 1.142 or obj_y < -0.392 or obj_y > 0.392:
                    return False
            return True
        else:
            return True


    def get_obj_pos(self):
        # get objects' z-axis height
        max_z = 0
        cube_up = None
        cube_x = None
        cube_y = None
        Get_Pos = GetModelStateRequest()
        for i in self.cube_name:
            Get_Pos.model_name = i
            obj_z = self.get_oject_position_service(Get_Pos).pose.position.z
            if obj_z > max_z:
                max_z = obj_z
                cube_up = i
                cube_x = self.get_oject_position_service(Get_Pos).pose.position.x
                cube_y = self.get_oject_position_service(Get_Pos).pose.position.y

        return [cube_up, max_z, cube_x, cube_y]


    def mygrasp(self, primitive_position, heightmap_rotation_angle, workspace_limits, up):
        # grasp process in Gazebo
        self.Moveit_Controller([primitive_position])
        self.gripper(action=[0])
        self.gripper_closed = False
        rospy.sleep(0.01)

        # controll end effortor to rotate 
        if heightmap_rotation_angle > np.pi:
            heightmap_rotation_angle = heightmap_rotation_angle - 2*np.pi
        tool_rotation_angle = heightmap_rotation_angle / 2
        self.end_joint_move(tool_rotation_angle)
        rospy.sleep(0.01)

        # move down to grasp
        grasp_down = [primitive_position[0], primitive_position[1], primitive_position[2] - 0.085 + up - 0.015]
        self.Moveit_Controller([grasp_down])
        rospy.sleep(0.01)

        # close the gripper(change me)
        self.gripper(action=[0.55])
        self.gripper_closed = True
        rospy.sleep(0.1)

        # back to the initial state
        self.initial_arm()
        rospy.sleep(0.01)

        # judge grasp one object or not
        grasp_success = False

        up_cube, max_z, cube_x, cube_y = self.get_obj_pos()
        gripper_x, gripper_y = primitive_position[0], primitive_position[1]
        if self.gripper_closed and max_z >= 0.80:
            grasp_success = True

        if grasp_success:
            self.gripper(action=[0])
            self.gripper_closed = False
            rospy.sleep(0.08)
            self.out_obj_pos(up_cube)
            self.cube_name.remove(up_cube)

        num_in_workspace = len(self.cube_name)

        return grasp_success, num_in_workspace


    def realgrasp(self, primitive_position, heightmap_rotation_angle, workspace_limits, down):
        # grasp process in real world
        self.move(primitive_position)
        if self.gripper_closed:
            self.real_gripper(100)
            self.gripper_closed = False
        time.sleep(0.01)

        # controll end effortor to rotate
        if heightmap_rotation_angle > np.pi:
            heightmap_rotation_angle = heightmap_rotation_angle - 2 * np.pi
        tool_rotation_angle = heightmap_rotation_angle / 2
        self.real_end_joint_move(tool_rotation_angle)

        # grasp
        push_down = [primitive_position[0], primitive_position[1], down]
        self.move(push_down)
        time.sleep(0.01)

        self.real_gripper(0)
        self.gripper_closed = True
        time.sleep(0.08)

        up_arm = [push_down[0], push_down[1], 0.647]
        self.move(up_arm)
        time.sleep(0.01)

        self.move_home()
        rospy.sleep(0.01)

        # judge grasp one object or not
        grasp_success = False
        gripper_pos = self.get_gripper_pose(1).return_data
        if self.gripper_closed and gripper_pos > 5:
            # put obj to the box
            self.move([-0.540, None, 0.640])
            time.sleep(0.01)

            self.real_gripper(100)
            self.gripper_closed = False

            self.move_home()
            time.sleep(0.01)
            grasp_success = True
        else:
            pass
        
        return grasp_success


    def mypush(self,primitive_position, heightmap_rotation_angle, workspace_limits,down):
        # push process in Gazebo
        self.Moveit_Controller([primitive_position])
        self.gripper(action=[0.8])
        self.gripper_closed = True
        rospy.sleep(0.01)

        # controll end effortor to rotate 
        tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi / 2
        self.end_joint_move(tool_rotation_angle)
        rospy.sleep(0.01)

        # the orientation of push
        push_orientation = [0.0, 1.0]
        push_direction = np.asarray([push_orientation[0]*np.cos(heightmap_rotation_angle) + push_orientation[1]*np.sin(heightmap_rotation_angle), 
                                                        push_orientation[0]*np.sin(heightmap_rotation_angle) + push_orientation[1]*np.cos(heightmap_rotation_angle)])

        # goal position
        push_length = 0.15
        target_x = min(max(primitive_position[0] + push_direction[0]*push_length, workspace_limits[0][0]), workspace_limits[0][1])
        target_y = min(max(primitive_position[1] + push_direction[1]*push_length, workspace_limits[1][0]), workspace_limits[1][1])

        # the distance to object , if pos_z == 0.97 ,we need to down about 0.045,else,if pos_z == 0.92,we need to down 0.085
        push_down = [primitive_position[0], primitive_position[1], primitive_position[2] - down - 0.005] 
        self.move_horizon(push_down)

        push_orient = [target_x, target_y, push_down[2]]
        self.move_horizon(push_orient)

        up_arm = [push_orient[0], push_orient[1], push_orient[2] + down]
        self.move_horizon(up_arm)

        self.initial_arm()
        rospy.sleep(0.01)

        push_success = True
        return push_success


    def realpush(self, primitive_position, heightmap_rotation_angle, workspace_limits, down):
        # push process in real world
        self.move(primitive_position)
        if not self.gripper_closed:
            self.real_gripper(0)
            self.gripper_closed = True
        time.sleep(0.01)
        
        # controll end effortor to rotate
        tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi / 2
        self.real_end_joint_move(tool_rotation_angle)
        time.sleep(0.01)

        # the orientation of push
        push_orientation = [0.0, 1.0]
        push_direction = np.asarray([-push_orientation[0]*np.cos(heightmap_rotation_angle) - push_orientation[1]*np.sin(heightmap_rotation_angle),
                                                        push_orientation[0]*np.sin(heightmap_rotation_angle) - push_orientation[1]*np.cos(heightmap_rotation_angle)])

        # goal position
        push_length = 0.17
        target_x = min(max(primitive_position[0] + push_direction[0]*push_length, workspace_limits[0][0]), workspace_limits[0][1])
        target_y = min(max(primitive_position[1] + push_direction[1]*push_length, workspace_limits[1][0]), workspace_limits[1][1])

        push_down = [primitive_position[0], primitive_position[1], down]
        self.move(push_down)
        time.sleep(0.01)

        push_orient = [target_x, target_y, push_down[2]]
        self.move(push_orient)
        time.sleep(0.01)

        up_arm = [push_orient[0], push_orient[1], 0.647]
        self.move(up_arm)
        time.sleep(0.01)

        self.real_reset()
        rospy.sleep(0.01)

        push_success = True
        return push_success


    def Moveit_Controller(self,move_pos):
        # controll the UR arm by cartesian in simulation
        execute_iter = 0
        self.arm.set_goal_position_tolerance(0.001)
        self.arm.set_goal_orientation_tolerance(0.01)
        self.arm.set_max_acceleration_scaling_factor(0.8)
        self.arm.set_max_velocity_scaling_factor(1)

        start_pose = self.arm.get_current_pose(self.end_effector_link).pose

        waypoints = []

        pos_num = len(move_pos)

        if pos_num > 1:
            for pos in move_pos:
                target_pose = PoseStamped()
                target_pose.header.frame_id = self.reference_frame
                target_pose.header.stamp = rospy.Time.now()     
                target_pose.pose.position.x = pos[0]
                target_pose.pose.position.y = pos[1]
                target_pose.pose.position.z = pos[2]
                target_pose.pose.orientation.x = start_pose.orientation.x
                target_pose.pose.orientation.y = start_pose.orientation.y
                target_pose.pose.orientation.z = start_pose.orientation.z
                target_pose.pose.orientation.w = start_pose.orientation.w
                waypoints.append(target_pose.pose)
        else:
            target_pose = PoseStamped()
            target_pose.header.frame_id = self.reference_frame
            target_pose.header.stamp = rospy.Time.now()     
            target_pose.pose.position.x = move_pos[0][0]
            target_pose.pose.position.y = move_pos[0][1]
            target_pose.pose.position.z = move_pos[0][2]
            target_pose.pose.orientation.x = start_pose.orientation.x
            target_pose.pose.orientation.y = start_pose.orientation.y
            target_pose.pose.orientation.z = start_pose.orientation.z
            target_pose.pose.orientation.w = start_pose.orientation.w
            waypoints.append(target_pose.pose)

        
        fraction = 0.0
        maxtries = 100
        attempts = 0 
        MPos_succ = False

        self.arm.set_start_state_to_current_state()

        while fraction < 1.0 and attempts < maxtries:
            (plan, fraction) = self.arm.compute_cartesian_path (
                                    waypoints,
                                    0.01,
                                    0.0,
                                True)
            attempts += 1         

        if fraction == 1.0:
            self.arm.execute(plan)
            MPos_succ = True
            rospy.sleep(0.2)
        else:
            rospy.loginfo("Path planning failed")
            self.initial_arm() 
        return MPos_succ


    def move_horizon(self, movePos):
        # controll the UR arm in simulation
        self.arm.set_max_acceleration_scaling_factor(0.24)
        self.arm.set_max_velocity_scaling_factor(0.48)
        start_pose = self.arm.get_current_pose(self.end_effector_link).pose

        target_pose = PoseStamped()
        target_pose.header.frame_id = self.reference_frame
        target_pose.header.stamp = rospy.Time.now()     
        target_pose.pose.position.x = (start_pose.position.x if movePos[0] == None else movePos[0])
        target_pose.pose.position.y = (start_pose.position.y if movePos[1] == None else movePos[1])
        target_pose.pose.position.z = (start_pose.position.z if movePos[2] == None else movePos[2])
        target_pose.pose.orientation.x = start_pose.orientation.x
        target_pose.pose.orientation.y = start_pose.orientation.y
        target_pose.pose.orientation.z = start_pose.orientation.z
        target_pose.pose.orientation.w = start_pose.orientation.w

        self.arm.set_start_state_to_current_state()
        self.arm.set_pose_target(target_pose, self.end_effector_link)
        trajectory = self.arm.plan()
        out = self.arm.execute(trajectory)
        rospy.sleep(0.1)


    def move(self, movePos):
        # # controll the UR arm in real world
        self.arm.set_max_acceleration_scaling_factor(0.24)
        self.arm.set_max_velocity_scaling_factor(0.48) 
        start_pose = self.arm.get_current_pose(self.end_effector_link).pose

        target_pose = PoseStamped()
        target_pose.header.frame_id = self.reference_frame
        target_pose.header.stamp = rospy.Time.now()     
        target_pose.pose.position.x = (start_pose.position.x if movePos[0] == None else movePos[0])
        target_pose.pose.position.y = (start_pose.position.y if movePos[1] == None else movePos[1])
        target_pose.pose.position.z = (start_pose.position.z if movePos[2] == None else movePos[2])
        target_pose.pose.orientation.x = start_pose.orientation.x
        target_pose.pose.orientation.y = start_pose.orientation.y
        target_pose.pose.orientation.z = start_pose.orientation.z
        target_pose.pose.orientation.w = start_pose.orientation.w

        self.arm.set_start_state_to_current_state()
        self.arm.set_pose_target(target_pose, self.end_effector_link)
        trajectory = self.arm.plan()
        out = self.arm.execute(trajectory)
        rospy.sleep(0.1)


    def end_joint_move(self,rad):
        # controll the rotation of the end effortor in Gazebo
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = JOINT_NAMES
        try:
            joint_states = rospy.wait_for_message("arm_controller/state", JointTrajectoryControllerState)
            joints_pos = list(joint_states.actual.positions)
            action_sent = [joints_pos[0], joints_pos[1], joints_pos[2], joints_pos[3], joints_pos[4], joints_pos[5] + rad]
            goal.trajectory.points = [
                JointTrajectoryPoint(positions=action_sent, velocities=[0]*6, time_from_start=rospy.Duration(1)),]
            self.client.send_goal(goal)
            self.client.wait_for_result()
        except KeyboardInterrupt:
            self.client.cancel_goal()
            raise


    def real_end_joint_move(self, rad):
        # controll the rotation of the end effortor in Real world
        self.arm.set_max_acceleration_scaling_factor(0.5)
        self.arm.set_max_velocity_scaling_factor(0.8)
        joint_value = self.arm.get_current_joint_values()
        joint_value[5] = joint_value[5] + rad
        self.arm.set_joint_value_target(joint_value)
        self.arm.go()
        rospy.sleep(0.1)


    def move_home(self):
        # back to the inital state
        joint_positions = [np.pi/2, -105*np.pi/180, 80*np.pi/180, -65*np.pi/180, -np.pi/2, np.pi/2]
        self.arm.set_joint_value_target(joint_positions)
        self.arm.go()
        rospy.sleep(0.1)


