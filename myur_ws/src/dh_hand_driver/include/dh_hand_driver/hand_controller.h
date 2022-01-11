#include <ros/ros.h>
#include <serial/serial.h>
#include <actionlib/server/simple_action_server.h>
#include <dh_hand_driver/definition.h>
#include <dh_hand_driver/ActuateHandAction.h>
#include <dh_hand_driver/hand_state.h>
#include "dh_hand_driver/hand_driver.h"
#include <mutex>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <arpa/inet.h>

class HandController
{
private:
    /// actionlib Server
    actionlib::SimpleActionServer<dh_hand_driver::ActuateHandAction> as_;

    /// Serial ports instance
    serial::Serial hand_ser_;

    /// controllers for the individual servo motors
    std::string hand_port_name_;

    /// DH Hand Model
    std::string Hand_Model_;

    // Wait data time;
    double WaitDataTime_;

    //connect mode 
    int connect_mode;

    //socket id
    int sockfd;

public:
    HandController(ros::NodeHandle n,const std::string& name);
    ~HandController();
 
    std::mutex R_mutex;
    std::mutex W_mutex;
    DH_Hand_Base Hand;
    DH_Robotics::DH_DataStream readtempdata;


    /**
     * @brief Callback for the action.
     * 
     * @param goal 
     */
    void actuateHandCB(const dh_hand_driver::ActuateHandGoalConstPtr &goal);
    
    /**
     * @brief Open DH_Hand communication
     * 
     * @return true on success; otherwise returns false. 
     */
    bool build_conn();

    /**
     * @brief Close DH_Hand communication
     * 
     * @return true on success; otherwise returns false. 
     */
    void closeDevice();

    /**
     * @brief Initialize the Hand
     * 
     */
    bool initHand();

    /**
     * @brief Move individual joint
     * 
     * @param MotorID 
     * @param target_position 
     * @return true on success; otherwise returns false. 
     */
    bool moveHand(int MotorID ,int target_position,bool &feedback);

    /**
     * @brief Set the Gripping Force
     * 
     * @param gripping_force 
     */
    bool setGrippingForce(int gripping_force);

    /**
     * @brief Write data to serial
     * 
     * @param data 
     * @return true 
     * @return false 
     */
    bool Writedata(std::vector<uint8_t> data);

    /**
     * @brief read recevied data ,save to tempdatastream
     * 
     * @return true on recevied success; 
     * @return false on recevied overtime;
     */
    bool Readdata(uint8_t waitconunt = 5);

    bool ensure_set_command(std::vector<uint8_t> data);
    bool ensure_get_command(std::vector<uint8_t> data);
    bool ensure_run_end(std::vector<uint8_t> data);
    bool chacke_data(uint8_t *data);

    /**
     * @brief service callback function
     * 
     * @param req 
     * @param res 
     * @return true on command is vaild;otherwise return false
     */
    bool jointValueCB(dh_hand_driver::hand_state::Request  &req,
         dh_hand_driver::hand_state::Response &res);

};