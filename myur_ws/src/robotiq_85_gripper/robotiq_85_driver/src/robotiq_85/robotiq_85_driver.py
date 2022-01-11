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
 
 \file   robotiq_85_driver.py

 \brief  Driver for Robotiq 85 communication

 \Platform: Linux/ROS Indigo
--------------------------------------------------------------------"""
from robotiq_85_gripper import Robotiq85Gripper
from robotiq_85_msgs.msg import GripperCmd, GripperStat
from sensor_msgs.msg import JointState
import numpy as np
import rospy

class Robotiq85Driver:
    def __init__(self):
        self._num_grippers = rospy.get_param('~num_grippers',1)
        self._comport = rospy.get_param('~comport','/dev/ttyUSB0')
        self._baud = rospy.get_param('~baud','115200')

        self._gripper = Robotiq85Gripper(self._num_grippers,self._comport,self._baud)

        if not self._gripper.init_success:
            rospy.logerr("Unable to open commport to %s" % self._comport)
            return
            
        if (self._num_grippers == 1):
            rospy.Subscriber("/gripper/cmd", GripperCmd, self._update_gripper_cmd, queue_size=10)
            self._gripper_pub = rospy.Publisher('/gripper/stat', GripperStat, queue_size=10)
            self._gripper_joint_state_pub = rospy.Publisher('/gripper/joint_states', JointState, queue_size=10)        
        elif (self._num_grippers == 2):
            rospy.Subscriber("/left_gripper/cmd", GripperCmd, self._update_gripper_cmd, queue_size=10)
            self._left_gripper_pub = rospy.Publisher('/left_gripper/stat', GripperStat, queue_size=10)
            self._left_gripper_joint_state_pub = rospy.Publisher('/left_gripper/joint_states', JointState, queue_size=10)
            rospy.Subscriber("/right_gripper/cmd", GripperCmd, self._update_right_gripper_cmd, queue_size=10)
            self._right_gripper_pub = rospy.Publisher('/right_gripper/stat', GripperStat, queue_size=10)
            self._right_gripper_joint_state_pub = rospy.Publisher('/right_gripper/joint_states', JointState, queue_size=10)
        else:
            rospy.logerr("Number of grippers not supported (needs to be 1 or 2)")
            return

        self._seq = [0] * self._num_grippers
        self._prev_js_pos = [0.0] * self._num_grippers
        self._prev_js_time = [rospy.get_time()] * self._num_grippers 
        self._driver_state = 0
        self._driver_ready = False
        
        success = True
        for i in range(self._num_grippers):
            success &= self._gripper.process_stat_cmd(i)
            if not success:
                bad_gripper = i
        if not success:
            rospy.logerr("Failed to contact gripper %d....ABORTING"%bad_gripper)
            return                
                
        self._run_driver()
        
    def _clamp_cmd(self,cmd,lower,upper):
        if (cmd < lower):
            return lower
        elif (cmd > upper):
            return upper
        else:
            return cmd

    def _update_gripper_cmd(self,cmd):
    
        if (True == cmd.emergency_release):
            self._gripper.activate_emergency_release(open_gripper=cmd.emergency_release_dir)
            return
        else:
            self._gripper.deactivate_emergency_release()

        if (True == cmd.stop):
            self._gripper.stop()
        else:
            pos = self._clamp_cmd(cmd.position,0.0,0.085)
            vel = self._clamp_cmd(cmd.speed,0.013,0.1)
            force = self._clamp_cmd(cmd.force,5.0,220.0)
            self._gripper.goto(dev=0,pos=pos,vel=vel,force=force)
            
    def _update_right_gripper_cmd(self,cmd):
    
        if (True == cmd.emergency_release):
            self._gripper.activate_emergency_release(dev=1,open_gripper=cmd.emergency_release_dir)
            return
        else:
            self._gripper.deactivate_emergency_release(dev=1)

        if (True == cmd.stop):
            self._gripper.stop(dev=1)
        else:
            pos = self._clamp_cmd(cmd.position,0.0,0.085)
            vel = self._clamp_cmd(cmd.speed,0.013,0.1)
            force = self._clamp_cmd(cmd.force,5.0,220.0)
            self._gripper.goto(dev=1,pos=pos,vel=vel,force=force)
            
    def _update_gripper_stat(self,dev=0):
        stat = GripperStat()
        stat.header.stamp = rospy.get_rostime()
        stat.header.seq = self._seq[dev]
        stat.is_ready = self._gripper.is_ready(dev)
        stat.is_reset = self._gripper.is_reset(dev)
        stat.is_moving = self._gripper.is_moving(dev)
        stat.obj_detected = self._gripper.object_detected(dev)
        stat.fault_status = self._gripper.get_fault_status(dev)
        stat.position = self._gripper.get_pos(dev)
        stat.requested_position = self._gripper.get_req_pos(dev)
        stat.current = self._gripper.get_current(dev)
        self._seq[dev]+=1
        return stat
        
    def _update_gripper_joint_state(self,dev=0):
        js = JointState()
        js.header.frame_id = ''
        js.header.stamp = rospy.get_rostime()
        js.header.seq = self._seq[dev]
        js.name = ['gripper_finger1_joint']
        pos = np.clip(0.8 - ((0.8/0.085) * self._gripper.get_pos(dev)), 0., 0.8)
        js.position = [pos]
        dt = rospy.get_time() - self._prev_js_time[dev]
        self._prev_js_time[dev] = rospy.get_time()
        js.velocity = [(pos-self._prev_js_pos[dev])/dt]
        self._prev_js_pos[dev] = pos
        return js
        
    def _run_driver(self):
        last_time = rospy.get_time()
        r = rospy.Rate(100)
        while not rospy.is_shutdown():
            dt = rospy.get_time() - last_time
            if (0 == self._driver_state):
                for i in range(self._num_grippers):
                    if (dt < 0.5):
                        self._gripper.deactivate_gripper(i)
                    else:
                        self._driver_state = 1
            elif (1 == self._driver_state):
                grippers_activated = True
                for i in range(self._num_grippers):    
                    self._gripper.activate_gripper(i)
                    grippers_activated &= self._gripper.is_ready(i)
                if (grippers_activated):
                    self._driver_state = 2
            elif (2 == self._driver_state):
                self._driver_ready = True
                        
            for i in range(self._num_grippers):
                success = True
                success &= self._gripper.process_act_cmd(i)
                success &= self._gripper.process_stat_cmd(i)
                if not success:
                    rospy.logerr("Failed to contact gripper %d"%i)
                
                else:
                    stat = GripperStat()
                    js = JointState()
                    stat = self._update_gripper_stat(i)
                    js = self._update_gripper_joint_state(i)
                    if (1 == self._num_grippers):
                        self._gripper_pub.publish(stat)
                        self._gripper_joint_state_pub.publish(js)
                    else:
                        if (i == 0):
                            self._left_gripper_pub.publish(stat)
                            self._left_gripper_joint_state_pub.publish(js)
                        else:
                            self._right_gripper_pub.publish(stat)
                            self._right_gripper_joint_state_pub.publish(js)
                            
            r.sleep()

        self._gripper.shutdown()
