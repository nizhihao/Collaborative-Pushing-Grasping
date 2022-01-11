# DH hand controller 

### Description
This is the control interface for the DH hand. 

### Software Architecture
The controller use ROS actionlib interface to control the displacement of joint associated hand. 

------



### Directory Structure

```
dh_hand_driver/
    ├── CMakeLists.txt
    ├── package.xml
    ├── README.md						# This file
    ├── action/
    │   └── ActuateHand.action			# To generate action header file
    ├── doc/
    │   └── <RegisterList>.xlsx			# List of Register 
    ├── include/
    │   └── dh_hand_driver/
    │       ├── definition.h			# All defintion of enum
    │       ├── DH_datastream.h			# Data Structure Class
    │       ├── hand_controller.h		# Main Controller Class
    │       └── hand_driver.h			# Process data Structure
    ├── launch/
    │   └── dh_hand_controller.launch	# Controller launch file
    ├── lib/
    │   └──libdh_hand_driver.so			# Dynamic Link Library
    ├── scripts/						# rule scripts
    │   ├── create_hand_rules.sh		
    │   ├── delete_hand_rules.sh
    │   └── DH_hand.rules
    ├── src/
    │   ├── node.cpp					# Controller Main Node
    │   ├── test_client.cpp				# Test hand example
    └── srv/  
		└── hand_state.srv           	# To genrate servise header file
```

------



### Installation

1. Make sure you have already installed and Configured Your [ROS](http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment)  Environment

2. Create and build a [catkin workspace](http://wiki.ros.org/catkin/workspaces): (you can skip this step , if you have already created it)

   ```
   $ mkdir -p ~/catkin_ws/src
   $ cd ~/catkin_ws/
   $ catkin_make
   ```

3. Install Dependencies :

   ```
   $ sudo apt-get install ros-<distro>-serial
   ```

   Using the short name of your ROS distribution instead of `<distro>`

   If you installed ROS Kinetic, that would be:

   ```
   $ sudo apt-get install ros-kinetic-serial
   ```

   or you can visit [Serial](https://github.com/wjwwood/serial) on Github to install the driver. 

4. Clone the whole directory( `dh_hand_driver` ) to  your catkin's workspace src folder

5. Compile the code 

   ```
   $ cd ~/catkin_ws/
   $ catkin_make
   ```

   > Notice: If compile error : dh_hand_driver/hand_state.h: No such file or directory 
   > 		please compile it again
   >
   > Or try : catkin_make -j1

6. Add the path including to the `ROS_PACKAGE_PATH` environment variable. Open `~/.bashrc` file and add at the end the following line. 

   ```
   source ~/catkin_ws/devel/setup.bash
   ```

8. Remap Serial port

   ```
   $ cd ~/catkin_ws/src/dh_hand_driver/
   $ ./scripts/create_udev_rules.sh
   ```

------



### Instructions

1. ##### Connect all hardware and Turn on the power

   Then check the remap using following command : `$ ls -l /dev | grep ttyACM`

   You will see output such as the following 

   ```
   lrwxrwxrwx  1 root root           7 9月   5 15:01 DH_hand -> ttyACM0
   crwxrwxrwx  1 root dialout 166,   0 9月   5 15:01 ttyACM0
   ```

   After you have change the Serial port remap, you can change the launch file about the serial_port value:(this is default)
   	`<param name="serial_port" type="string" value="/dev/DH_hand"/>`

   otherwise  change the launch file (such as /dev/ttyACM0) :
   	`<param name="serial_port" type="string" value="/dev/ttyACM0 "/>`

      ​	And add the authority of write :`$ sudo chmod 666 /dev/ttyACM0`

2. ##### Modify launch file and Running controller

   First , Modify the  `dh_hand_controller.launch` file in according to the product model (`AG-2E` or `AG-3E`)
   	`<param name="Hand_Model" type="string" value="AG-2E"/>`
   Now , you can running controller

   ```
   $ roslaunch dh_hand_driver dh_hand_controller.launch		
   ```

   > If If it runs successfully , you will see the initialization of the hand

3. ##### Run the  client

   after you run the dh_hand_controller.launch , you can run the `hand_controller_client` to control it

   ```
   $ rosrun dh_hand_driver hand_controller_client [motorID][position][Force]
   ```

   > motorID	: 	AG-2E has one motor ,  this parameter just be 1;
   >				AG-3E has two motor ,  this parameter can  be 1 or 2
   >
   > position	:	Range from 0   to 100
   >
   > Force	: 	Range from 20 to 100

   such as :

   ```
   $ rosrun dh_hand_driver hand_controller_client 1 50 90
   ```

4. ##### Study how to use it 

   you can read the ` test_client.cpp`  in `/dh_hand_driver/src` folder to study

   we use the [actionlib](http://wiki.ros.org/actionlib) to send Goal and get Feedback and Result:

   > Goal		: motorID , position and Force
   > 
   > Feedback	: Whether to reach the target Position
   > 
   > Result		: Whether successful execution

   And we use the [service](http://wiki.ros.org/Services) to get the hand state:

   > get Force		: = 0
   > 
   > get Position 1 	: = 1
   > 
   > get Position 2	: = 2

5. ##### Enjoy it

6. ## TODO:
	> Add URDF support and simulation
	>
	> More control interfaces
