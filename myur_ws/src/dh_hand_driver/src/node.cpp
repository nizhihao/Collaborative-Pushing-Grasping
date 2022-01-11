#include <ros/ros.h>
#include <dh_hand_driver/hand_controller.h>
#include <dh_hand_driver/Pos_state.h>
#include <stdio.h>
#include <iostream>
using namespace std;

int main(int argc, char** argv)
{
  ros::init(argc, argv, "hand_controller");
  ros::NodeHandle n;
  cout << '1' << endl;
  HandController controller(n,"actuate_hand");
  cout << '2' << endl;
  ros::spin();

  return 0;
}

