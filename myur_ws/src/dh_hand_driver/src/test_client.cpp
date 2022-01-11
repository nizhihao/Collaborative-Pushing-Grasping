#include <ros/ros.h>
#include <dh_hand_driver/ActuateHandAction.h>
#include <dh_hand_driver/hand_state.h>
#include <actionlib/client/simple_action_client.h>

typedef actionlib::SimpleActionClient<dh_hand_driver::ActuateHandAction> Client;

class DH_HandActionClient {
private:
    // Called once when the goal completes
    void DoneCb(const actionlib::SimpleClientGoalState& state,
            const dh_hand_driver::ActuateHandResultConstPtr& result) {
        ROS_INFO("Finished in state [%s]", state.toString().c_str());
        ROS_INFO("result  : %i", result->opration_done);
    }

    // when target active, call this once
    void ActiveCb() {
        ROS_INFO("Goal just went active");
    }

    // received feedback
    void FeedbackCb(
            const dh_hand_driver::ActuateHandFeedbackConstPtr& feedback) {
        ROS_INFO("Got Feedback: %i", feedback->position_reached);
    }
public:
    DH_HandActionClient(const std::string client_name, bool flag = true) :
            client(client_name, flag) {
    }

    //client start
    void Start(int32_t motorID , int32_t setpos,int32_t setforce) {
        ROS_INFO("wait server");
        client.waitForServer();
        //set goal
        dh_hand_driver::ActuateHandGoal goal;
        // AG2E just has one motor (ID:1)
        // AG3E has two motor (ID:1 and 2)
        goal.MotorID    = motorID;
        goal.force      = setforce;
        goal.position   = setpos;

        ROS_INFO("Send goal %d %d %d", goal.MotorID,goal.force,goal.position);
        //sen goal
        client.sendGoal(goal,
                boost::bind(&DH_HandActionClient::DoneCb, this, _1, _2),
                boost::bind(&DH_HandActionClient::ActiveCb, this),
                boost::bind(&DH_HandActionClient::FeedbackCb, this, _1));
        ROS_INFO("wait result");
 
        client.waitForResult(ros::Duration(15.0));

        //process the result
        if (client.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
            ROS_INFO("Send commond succeeded");
        else {
            ROS_WARN("Cancel Goal!");
            client.cancelAllGoals();
        }

        printf("Current State: %s\n", client.getState().toString().c_str());
    }
private:
    Client client;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "test_hand_client");
    if(argc != 4)
    {
         ROS_INFO("Useage: rosrun dh_hand_driver hand_controller_client [motorID] [position] [Force]! ");
            return -1;
    }
  ROS_INFO("starting");
  ros::NodeHandle n;
  DH_HandActionClient actionclient("actuate_hand", true);

     //use service to get hand state
    ros::ServiceClient client = n.serviceClient<dh_hand_driver::hand_state>("hand_joint_state");
    dh_hand_driver::hand_state srv;

//   int i_position=0;
//   int i_force=20;
  while(1)
  {

    // if(i_position>100)
    // {
    //     i_position=0;
    // }    
    // if(i_force>100)
    // {
    //     i_force=20;
    // }   
    // i_position++;
    // i_force++;  
     ROS_INFO("starting client");
    actionclient.Start(atoi(argv[1]),atoi(argv[2]),atoi(argv[3]));
    // actionclient.Start(1,i_position,i_force);
    ros::spinOnce();


      srv.request.get_target = 0;
    if (client.call(srv))
    {
        ROS_INFO("force: %d",srv.response.return_data);
    }
     else
    {
        ROS_ERROR("Failed to call service");
        return 1;
    }
    srv.request.get_target = 1;
    if (client.call(srv))
    {
        ROS_INFO("pos_1: %d",srv.response.return_data);
    }
     else
    {
        ROS_ERROR("Failed to call service");
        return 1;
    }
  
    actionclient.Start(1,100,100);
    ros::spinOnce();


//    //use service to get hand state
//     ros::ServiceClient client = n.serviceClient<dh_hand_driver::hand_state>("hand_joint_state");
//     dh_hand_driver::hand_state srv;
    srv.request.get_target = 0;
    if (client.call(srv))
    {
        ROS_INFO("force: %d",srv.response.return_data);
    }
     else
    {
        ROS_ERROR("Failed to call service");
        return 1;
    }
    srv.request.get_target = 1;
    if (client.call(srv))
    {
        ROS_INFO("pos_1: %d",srv.response.return_data);
    }
     else
    {
        ROS_ERROR("Failed to call service");
        return 1;
    }
}
    //this command belong AG-3E
    //    srv.request.get_target = 2;
    // if (client.call(srv))
    // {
    //     ROS_INFO("pos_2: %d",srv.response.return_data);
    // }
    //  else
    // {
    //     ROS_ERROR("Failed to call service");
    //     return 1;
    // }
    ros::shutdown();
    return 0;
}
