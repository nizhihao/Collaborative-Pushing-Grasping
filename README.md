# Collaborative Pushing and Grasping of Tightly Stacked Objects via Deep Reinforcement Learning

**Abstract**—Directly grasping the tightly stacked objects may cause collisions and result in failures, degenerating the functionality of robotic arms. Inspired by the observation that first pushing objects to a state of mutual separation and then grasping them individually can effectively increase the success rate, we devise a novel deep Q-learning framework to achieve collaborative pushing and grasping. Specifically, an efficient nonmaximum suppression policy (PolicyNMS) is proposed to dynamically evaluate pushing and grasping actions by enforcing a suppression constraint on unreasonable actions. Moreover, a novel data driven pushing reward network called PR-Net is designed to effectively assess the degree of separation or aggregation between objects. To benchmark the proposed method, we establish a dataset containing common household items dataset (CHID) in both simulation and real scenarios. Although trained using simulation data only, experiment results validate that our method generalizes well to real scenarios and achieves a 97% grasp success rate at a fast speed for object separation in the real-world environment.

[PDF](https://www.ieee-jas.net/en/article/doi/10.1109/JAS.2021.1004255) | [Video Results](https://github.com/nizhihao/Collaborative-Pushing-Grasping/tree/master/video)

## Installation

This implementation requires the following dependencies (tested on Ubuntu 16.04 LTS):

- Python 2.7. The default python version in the ubuntu 16.04.

- [ROS Kinetic](http://wiki.ros.org/Installation/Ubuntu). You can quickly install the ROS and Gazebo by following the wiki installation web(if you missing dependency package, you can install these package by runing the following):

  ```shell
  sudo apt-get install ros-kinetic-(Name)
  sudo apt-get install ros-kinetic-(the part of Name)*   # * is the grammar of regular expression
  ```

- [NumPy](http://www.numpy.org/), [SciPy](https://www.scipy.org/scipylib/index.html), [OpenCV-Python](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html), [Matplotlib](https://matplotlib.org/). You can quickly install/update these dependencies by running the following:

  ```shell
  pip install numpy scipy opencv-python matplotlib
  ```

- Torch == 1.0.0 and Torchvision == 0.2.1

  ```shell
  pip install torch==1.0.0 torchvision==0.2.1
  ```

- CUDA and cudnn. You need to install the GPU driver、the cuda and the cudnn, this code has been tested with CUDA 9.0 and cudnn 7.1.4 on two 1080Ti GPU(11GB).

## Train or Test the algorithm in the Simulation(Gazebo)

1. download this repository and compile the ROS workspace.

   ```shell
   git clone https://github.com/nizhihao/Collaborative-Pushing-Grasping.git
   mv /home/user/Collaborative-Pushing-Grasping/myur_ws /home/user
   cd /home/user/myur_ws
   catkin_make -j1
   echo "source /home/user/myur_ws/devel/setup.bash" >> ~/.bashrc
   source ~/.bashrc
   ```

   The Introduction of the ROS package.

   ```shell
   dh_hand_driver   	# dh gripper's driver
   drl_push_grasp    # the RL package(all algorithm code in this package)
   gazebo-pkgs       # Gazebo grasp plugin
   robotiq_85_gripper   # robotiq_85 gripper package
   universal_robot-kinetic-devel  	# UR robotic arm package
   ur_modern_driver 		     # UR robotic arm driver in the real world 
   ur_robotiq 			  # URDF, Objects Mesh, Initial Env, MoveIt config package
   ```

2. If you want to train in Gazebo,  You can run the following code and set is_sim=True, is_testing=False in the main.py. 

   If you want to test in Gazebo,  You can run the following code and set is_sim=True, is_testing=True in the main.py. 

   ```
   Tips: 
   1.this repository only provides the training or testing process about Collaborative pushing grasping method, the training process of pushing policy and grasping policy don't export in this repository.
   2.You need to open a new Terminal for each command Line.
   ```

   ```shell
   roslaunch ur_robotiq_gazebo ur_robotiq_gazebo.launch   # run the Gazebo and MoveIt node
   rosrun drl_push_grasp main.py   # run the agent
   ```

   The Introduction of the drl_push_grasp(RL) package.

   ```shell
   logger.py      # save the log
   main.py   	   # main func
   multi_model.py  # network structure
   trainer.py   	 # agent 
   ur_robotiq.py  # environment
   utils.py		   # some function module
   ```

   

3. when finish the train or test, run the following code to draw the performance curve.

   ```
   cd /home/user/myur_ws/src/drl_push_grasp/scripts/
   # only compare the push or grasp policy
   python plot_ablation_push.py '../logs/YOUR-SESSION-DIRECTORY-NAME-HERE-01' '../logs/YOUR-SESSION-DIRECTORY-NAME-HERE-02'
   python plot_ablation_grasp.py '../logs/YOUR-SESSION-DIRECTORY-NAME-HERE-01' '../logs/YOUR-SESSION-DIRECTORY-NAME-HERE-02'
   # To plot the performance of pushing-grasping policy over training time
   python plot.py '../logs/YOUR-SESSION-DIRECTORY-NAME-HERE'
   ```

   

## Running on a Real Robot (UR10)

The same code in this repository can be used to test on a real UR10 robot arm (controlled with ROS). 

### Setting Up Camera System

The latest version of our system uses RGB-D data captured from an Intel® RealSense™ D435 Camera. We provide a lightweight C++ executable that streams data in real-time using [librealsense SDK 2.0](https://github.com/IntelRealSense/librealsense) via TCP. This enables you to connect the camera to an external computer and fetch RGB-D data remotely over the network while training. This can come in handy for many real robot setups. Of course, doing so is not required -- the entire system can also be run on the same computer.

#### Installation Instructions:

1. Download and install [librealsense SDK 2.0](https://github.com/IntelRealSense/librealsense)

1. Navigate to `drl_push_grasp/scripts/realsense` and compile `realsense.cpp`:

   ```shell
   cd /home/user/myur_ws/src/drl_push_grasp/scripts/realsense
   cmake .
   make
   ```

1. Connect your RealSense camera with a USB 3.0 compliant cable (important: RealSense D400 series uses a USB-C cable, but still requires them to be 3.0 compliant to be able to stream RGB-D data).

1. To start the TCP server and RGB-D streaming, run the following:

   ```shell
   ./realsense
   ```

### Calibrating Camera Extrinsics

We provide a simple calibration script to estimate camera extrinsics with respect to robot base coordinates. To do so, the script moves the robot gripper over a set of predefined 3D locations as the camera detects the center of a moving 4x4 checkerboard pattern taped onto the gripper. The checkerboard can be of any size (the larger, the better).

run the following to move the robot and calibrate:

```shell
cd /home/user/myur_ws/src/drl_push_grasp/scripts/realsense && ./realsense  # run the realsense to obtain the camera data
roslaunch ur_modern_driver ur10_bringup_joint_limited.launch robot_ip:=192.168.1.186 # run the ur10 arm
roslaunch ur10_moveit_config ur10_moveit_planning_#execution.launch  # run the MoveIt node
roslaunch dh_hand_driver dh_hand_controller.launch  # run the dh gripper
python calibrate_myrobot.py  # calibrate
```

### Testing

If you want to test in real world,  You can run the following code and set is_sim=False, is_testing=True in the main.py. 

```shell
cd /home/user/myur_ws/src/drl_push_grasp/scripts/realsense && ./realsense  # run the realsense to obtain the camera data
roslaunch ur_modern_driver ur10_bringup_joint_limited.launch robot_ip:=192.168.1.186 # run the ur10 arm
roslaunch ur10_moveit_config ur10_moveit_planning_execution.launch  # run the MoveIt node
roslaunch dh_hand_driver dh_hand_controller.launch  # run the dh gripper
rosrun drl_push_grasp main.py  # run the agent
```

## Citing

If you find this code useful in your work, please consider citing:

```shell
@articleInfo{JAS-2021-0389,
title = "Collaborative Pushing and Grasping of Tightly Stacked Objects via Deep Reinforcement Learning",
journal = "IEEE/CAA Journal of Automatica Sinica",
volume = "9",
number = "JAS-2021-0389,
pages = "135",
year = "2022",
note = "",
issn = "2329-9266",
doi = "10.1109/JAS.2021.1004255",
url = "https://www.ieee-jas.net//article/id/7a2afe68-eb96-4946-aa86-befec6b1fd66"
}
```

## Contact

If you have any questions or find any bugs, please let me know: ZhiHao Ni(nzh@hdu.edu.cn).

