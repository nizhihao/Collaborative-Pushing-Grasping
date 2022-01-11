"""--------------------------------------------------------------------
COPYRIGHT 2015 Stanley Innovation Inc.

Software License Agreement:

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 
 \file   robotiq_85_test.py

 \brief  Node for testing Robotiq 85 communication

 \Platform: Linux/ROS Indigo
--------------------------------------------------------------------"""
import rospy
from robotiq_85_msgs.msg import GripperCmd, GripperStat


class Robotiq85GripperTest:

    def __init__(self):
    
        self._num_grippers = rospy.get_param('~num_grippers',1)
        
        if (self._num_grippers == 1):
            rospy.Subscriber("/gripper/stat", GripperStat, self._update_gripper_stat, queue_size=10)
            self._gripper_pub = rospy.Publisher('/gripper/cmd', GripperCmd, queue_size=10)      
        elif (self._num_grippers == 2):
            rospy.Subscriber("/left_gripper/stat", GripperStat, self._update_gripper_stat, queue_size=10)
            self._left_gripper_pub = rospy.Publisher('/left_gripper/stat', GripperCmd, queue_size=10)
            rospy.Subscriber("/right_gripper/stat", GripperStat, self._update_right_gripper_stat, queue_size=10)
            self._right_gripper_pub = rospy.Publisher('/right_gripper/cmd', GripperCmd, queue_size=10)
        else:
            rospy.logerr("Number of grippers not supported (needs to be 1 or 2)")
            return
            
        self._gripper_stat = [GripperStat()] * self._num_grippers
        self._gripper_cmd = [GripperCmd()]  * self._num_grippers
            
        self._run_test()
        
        
    def _update_gripper_stat(self, stat):
        self._gripper_stat[0] = stat
    def _update_right_gripper_stat(self, stat):
        self._gripper_stat[1] = stat
    
    
    def _run_test(self):
        test_state = 0
        r = rospy.Rate(1)
        while not rospy.is_shutdown():
            ready = False
            while not (ready):
                ready = True
                for i in range(self._num_grippers):
                    ready &= self._gripper_stat[i].is_ready
            
            if (0 == test_state):
                for i in range(self._num_grippers):
                    self._gripper_cmd[i].position = 0.0
                    self._gripper_cmd[i].speed = 0.02
                    self._gripper_cmd[i].force = 100.0
                test_state = 1
            elif (1 == test_state):
                success = True
                for i in range(self._num_grippers):
                    if (self._gripper_stat[i].is_moving):
                        success = False
                if success:
                    test_state = 2                 

            if (2 == test_state):
                for i in range(self._num_grippers):
                    self._gripper_cmd[i].position = 0.085/3
                    self._gripper_cmd[i].speed = 0.02
                    self._gripper_cmd[i].force = 100.0
                test_state = 3
            elif (3 == test_state):
                success = True
                for i in range(self._num_grippers):
                    if (self._gripper_stat[i].is_moving):
                        success = False
                if success:
                    test_state = 4
            if (4 == test_state):
                for i in range(self._num_grippers):
                    self._gripper_cmd[i].position = 0.085/2
                    self._gripper_cmd[i].speed = 0.02
                    self._gripper_cmd[i].force = 100.0
                test_state = 5
            elif (5 == test_state):
                success = True
                for i in range(self._num_grippers):
                    if (self._gripper_stat[i].is_moving):
                        success = False
                if success:
                    test_state = 6
            if (6 == test_state):
                for i in range(self._num_grippers):
                    self._gripper_cmd[i].position = 0.085
                    self._gripper_cmd[i].speed = 0.02
                    self._gripper_cmd[i].force = 100.0
                test_state = 7
            elif (7 == test_state):
                success = True
                for i in range(self._num_grippers):
                    if (self._gripper_stat[i].is_moving):
                        success = False
                if success:
                    test_state = 0
                    
            if (self._num_grippers == 1):
                self._gripper_pub.publish(self._gripper_cmd[0])    
            elif (self._num_grippers == 2):
                self._left_gripper_pub.publish(self._gripper_cmd[0])
                self._right_gripper_pub.publish(self._gripper_cmd[1])
            
            r.sleep()                
                
            
                
        



