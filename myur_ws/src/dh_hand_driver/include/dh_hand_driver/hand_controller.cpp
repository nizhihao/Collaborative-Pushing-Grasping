/**
This is the control interface for the DH Hand
Author:   Jie Sun
Email:    jie.sun@dh-robotics.com
Date:     2019 July 1

Version 2.0
Copyright @ DH-Robotics Ltd.  
**/

#include <dh_hand_driver/hand_controller.h>
#include <dh_hand_driver/Pos_state.h>
HandController::HandController(ros::NodeHandle n, const std::string &name)
    : as_(n, name, boost::bind(&HandController::actuateHandCB, this, _1), false)
{
  n.param<std::string>("Connect_port", hand_port_name_, "/dev/DH_hand");
  n.param<std::string>("Hand_Model", Hand_Model_, "AG-2E");
  n.param<double>("WaitDataTime", WaitDataTime_, 0.5);  
  ROS_INFO("Hand_model : %s", Hand_Model_.c_str());
  ROS_INFO("Connect_port: %s", hand_port_name_.c_str());

  connect_mode = 0;
  if (hand_port_name_.find('/') != std::string::npos)
  {
    connect_mode = 1;
  }
  if (hand_port_name_.find(':') != std::string::npos)
  {
    connect_mode = 2;
  }
  ROS_INFO("connect_mode : %d", connect_mode);
  if (!build_conn())
  {
    return;
  }
  if (initHand())
  {
    ROS_INFO("Initialized");
  }
  ros::ServiceServer service = n.advertiseService("hand_joint_state", &HandController::jointValueCB, this);

  as_.start();
  ROS_INFO("server started");

 //消息类型数据发布
  //ros::NodeHandle node;
  //ros::Publisher pub = node.advertise<dh_hand_driver::Pos_state>("/Pos", 100);
  //dh_hand_driver::Pos_state msg;
  //readtempdata.DataStream_clear();
  //Hand.getMotorPosition(1);

  //msg.motor1_data = readtempdata.data[0];
  //ROS_INFO("Publish message is ok: position:%d", msg.motor1_data);
  //pub.publish(msg);

  ros::spin();
}

HandController::~HandController()
{
  closeDevice();
}

bool HandController::jointValueCB(dh_hand_driver::hand_state::Request &req,
                                  dh_hand_driver::hand_state::Response &res)
{
  auto ret = false;
  auto iter = 0;
  do
  {
    readtempdata.DataStream_clear();
    switch (req.get_target)
    {
    case 0:
      ROS_INFO("request: getMotorForce");
      Hand.getMotorForce();
      break;
    case 1:
      ROS_INFO("request: getMotor1Position");
      Hand.getMotorPosition(1);
      break;
    case 2:
      ROS_INFO("request: getMotor2Position");
      if (Hand_Model_ == "AG-3E")
      {
        Hand.getMotorPosition(2);
        break;
      }
    default:
      ROS_ERROR("invalid read command");
      return false;
      break;
    }


    if (Writedata(Hand.getStream()))
    {
      if (ensure_get_command(Hand.getStream()))
      {
        ret = true;
        break;
      }
      else
      {
        ROS_WARN("jointValueCB wait timeout");
      }
    }
    else
    {
      ROS_WARN("jointValueCB write command");
    }
  } while (iter++ < 100);

  res.return_data = readtempdata.data[0];
  return ret;
}

void HandController::actuateHandCB(const dh_hand_driver::ActuateHandGoalConstPtr &goal)
{
  ROS_INFO("Start to move the DH %s Hand", Hand_Model_.c_str());

  dh_hand_driver::ActuateHandFeedback feedback;
  dh_hand_driver::ActuateHandResult result;
  bool succeeded = false;
  bool bFeedback = false;

  // move Motor
  if (Hand_Model_ == "AG-2E" && goal->MotorID == 2)
  {
    ROS_ERROR("invalid AG-2E command");
    as_.setAborted(result);
    return;
  }

  if (goal->force >= 20 && goal->force <= 100 && goal->position >= 0 && goal->position <= 100)
  {
    setGrippingForce(goal->force);
    succeeded = moveHand(goal->MotorID, goal->position, bFeedback);

    feedback.position_reached = bFeedback;
    as_.publishFeedback(feedback);
    ros::Duration(WaitDataTime_).sleep();
    result.opration_done = succeeded;
  }
  else
  {
    ROS_ERROR("received invalid action command");
  }
  if (succeeded)
    as_.setSucceeded(result);
  else
    as_.setAborted(result);
}

/**
 * To build the communication with the servo motors
 *
* */
bool HandController::build_conn()
{

  bool hand_connected = false;
  if (connect_mode == 1)
  {
    try
    {
      hand_ser_.setPort(hand_port_name_);
      serial::Timeout to = serial::Timeout::simpleTimeout(1000);
      hand_ser_.setTimeout(to);
      hand_ser_.open();
    }
    catch (serial::IOException &e)
    {
      ROS_ERROR("Unable to open port of the hand");
      return hand_connected;
    }

    if (hand_ser_.isOpen())
    {
      ROS_INFO("Serial Port for hand initialized");
      hand_connected = true;
    }
    else
    {
      return hand_connected;
    }
  }
  else if (connect_mode == 2)
  {

    std::string servInetAddr = hand_port_name_.substr(0, hand_port_name_.find(":"));
    int PORT = atoi(hand_port_name_.substr(hand_port_name_.find(":") + 1, hand_port_name_.size() - hand_port_name_.find(":") - 1).c_str());

    /*创建socket*/
    struct sockaddr_in serv_addr;
    if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) != -1)
    {
      ROS_INFO("Socket id = %d", sockfd);
      /*设置sockaddr_in 结构体中相关参数*/
      serv_addr.sin_family = AF_INET;
      serv_addr.sin_port = htons(PORT);
      inet_pton(AF_INET, servInetAddr.c_str(), &serv_addr.sin_addr);
      bzero(&(serv_addr.sin_zero), 8);
      /*调用connect 函数主动发起对服务器端的连接*/
      if (connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) == -1)
      {
        ROS_ERROR("Connect failed!\n");
        hand_connected = false;
      }
      else
      {
        ROS_INFO("connected");
        hand_connected = true;
      }
    }
    else
    {
      ROS_ERROR("Socket failed!\n");
      hand_connected = false;
    }
  }
  return hand_connected;
}

void HandController::closeDevice()
{
  // stop communication
  if (connect_mode == 1)
    hand_ser_.close();
  else if (connect_mode == 2)
    close(sockfd);
}

bool HandController::initHand()
{
  auto ret = false;
  auto iter = 0;
  do
  {
    // gen initialize commond
    Hand.setInitialize();
    //write
    if (Writedata(Hand.getStream()))
    {
      // ensure set finished
      if (ensure_set_command(Hand.getStream()))
      {
        ROS_INFO("init ensure_set_command");
        //wait initialize finish
        Hand.setInitialize();
        auto wait_i = 0;
        do
        {
          ros::Duration(0.5).sleep();
          if (ensure_get_command(Hand.getStream()))
          {
            if (readtempdata.data[0] == 0x01)
              ret = true;
            break;
          }
        } while (wait_i++ < 20);

        if (ret)
          break;
      }
      else
      {
        ROS_WARN("setInitialize wait response timeout");
      }
    }
    else
    {
      ROS_WARN("setInitialize write data error");
    }
  } while (iter++ < 2);
  return ret;
}

bool HandController::moveHand(int MotorID, int target_position, bool &feedback)
{
  auto ret = false;
  auto iter = 0;
  auto wait_iter = 0;

  do
  {
    //gen position command frame
    Hand.setMotorPosition(MotorID, target_position);
    //write
    if (Writedata(Hand.getStream()))
    {
      // ensure set finished
      if (ensure_set_command(Hand.getStream()))
      {
        //gen get feedback frame
        Hand.getFeedback(MotorID);
        //write
        do
        {
          if (Writedata(Hand.getStream()))
          {
            //wait and ensure finished motion
            if (ensure_run_end(Hand.getStream()))
            {
              if (readtempdata.data[0] == DH_Robotics::FeedbackType::ARRIVED)
              {
                feedback = true;
                ret = true;
                break;
              }
              else if (readtempdata.data[0] == DH_Robotics::FeedbackType::CATCHED)
              {
                feedback = false;
                ret = true;
                break;
              }
            }
            else
            {
              ROS_WARN("getFeedback ensure_run_end wait response timeout");
            }
          }
          else
          {
            ROS_WARN("getFeedback wait response timeout");
          }
        } while (wait_iter++ < 100);
        // ROS_WARN("wait_iter %d", wait_iter);

      }
      else
      {
        ROS_WARN("setMotorPosition wait response timeout");
      }
    }
    else
    {
      ROS_WARN("setMotorPosition write data error");
    }
    if (ret)
      break;
  } while (iter++ < 3);
  return ret;
}

bool HandController::setGrippingForce(int gripping_force)
{
  auto ret = false;
  auto iter = 0;
  do
  {
    Hand.setMotorForce(gripping_force);
    if (Writedata(Hand.getStream()))
    {
      if (ensure_set_command(Hand.getStream()))
      {
        ret = true;
        break;
      }
      else
      {
        ROS_WARN("setMotorForce wait timeout");
      }
    }
    else
    {
      ROS_WARN("setMotorForce write data error");
    }
  } while (iter++ < 3);
  return ret;
}

bool HandController::Writedata(std::vector<uint8_t> data)
{
  bool ret = false;
  // ROS_INFO("send x");
  if (connect_mode == 1)
  {
    if (hand_ser_.write(data) == data.size())
    {
      ret = true;
    }
  }
  else if (connect_mode == 2)
  {
    if (write(sockfd, data.data(), data.size()) == (unsigned int)data.size())
    {
      ret = true;
    }
  }
  return ret;
}

bool HandController::ensure_set_command(std::vector<uint8_t> data)
{
  auto ret = false;
  auto iter = 0;
  do
  {
    if (Readdata(3))
    {
      if (data.at(0) == readtempdata.frame_head[0] &&
          data.at(1) == readtempdata.frame_head[1] &&
          data.at(2) == readtempdata.frame_head[2] &&
          data.at(3) == readtempdata.frame_head[3] &&
          data.at(4) == readtempdata.DeviceID &&
          data.at(5) == readtempdata.Register[0] &&
          data.at(6) == readtempdata.Register[1] &&
          data.at(7) == readtempdata.option &&
          data.at(8) == readtempdata.reserve &&
          data.at(9) == readtempdata.data[0] &&
          data.at(10) == readtempdata.data[1] &&
          data.at(11) == readtempdata.data[2] &&
          data.at(12) == readtempdata.data[3] &&
          data.at(13) == readtempdata.frame_end)
      {
        ret = true;
        break;
      }
    }
  } while (iter++ < 1);
  return ret;
}

bool HandController::ensure_get_command(std::vector<uint8_t> data)
{
  auto ret = false;
  auto iter = 0;
  do
  {
    if (Readdata(3))
    {
      if (data.at(0) == readtempdata.frame_head[0] &&
          data.at(1) == readtempdata.frame_head[1] &&
          data.at(2) == readtempdata.frame_head[2] &&
          data.at(3) == readtempdata.frame_head[3] &&
          data.at(4) == readtempdata.DeviceID &&
          data.at(5) == readtempdata.Register[0] &&
          data.at(6) == readtempdata.Register[1] &&
          // data.at(7) == (uint8_t)(0) &&
          data.at(8) == readtempdata.reserve &&
          // data.at(9) == readtempdata.data[0] &&
          data.at(10) == readtempdata.data[1] &&
          data.at(11) == readtempdata.data[2] &&
          data.at(12) == readtempdata.data[3] &&
          data.at(13) == readtempdata.frame_end)
      {
        ret = true;
        break;
      }
    }
  } while (iter++ < 1);
  return ret;
}

bool HandController::ensure_run_end(std::vector<uint8_t> data)
{
  auto ret = false;
  auto iter = 0;
  do
  {
    if (Readdata(3))
    {
      if (data.at(0) == readtempdata.frame_head[0] &&
          data.at(1) == readtempdata.frame_head[1] &&
          data.at(2) == readtempdata.frame_head[2] &&
          data.at(3) == readtempdata.frame_head[3] &&
          data.at(4) == readtempdata.DeviceID &&
          data.at(5) == readtempdata.Register[0] &&
          data.at(6) == readtempdata.Register[1] &&
          data.at(7) == readtempdata.option &&
          data.at(8) == readtempdata.reserve &&
          // 0 != readtempdata.data[0] &&
          data.at(10) == readtempdata.data[1] &&
          data.at(11) == readtempdata.data[2] &&
          data.at(12) == readtempdata.data[3] &&
          data.at(13) == readtempdata.frame_end)
      {
        ret = true;
        break;
      }
    }
  } while (iter++ < 1);
  return ret;
}

bool HandController::chacke_data(uint8_t *data)
{
  if (0xFF == data[0] &&
      0xFE == data[1] &&
      0xFD == data[2] &&
      0xFC == data[3] &&
      0xFB == data[13])
  {
    return true;
  }
  else
  {
    ROS_WARN("get data structure false");
    return false;
  }
}

bool HandController::Readdata(uint8_t waitconunt)
{
  auto ret = false;
  auto iter = 0;
  auto getframe = false;
  uint8_t buf[14];

  do
  {
    ros::Duration(WaitDataTime_).sleep();
    if (connect_mode == 1)
    {
      uint8_t count = hand_ser_.available() / 14;
      uint8_t remain = hand_ser_.available() % 14;
      if (count >= 1 && remain == 0)
      {
        for (; count > 1; count--)
        {
          hand_ser_.read(buf, 14);
        }
        hand_ser_.read(buf, 14);
        readtempdata.DatafromStream(buf, 14);
        getframe = true;
      }
    }
    else if (connect_mode == 2)
    {

      std::vector<uint8_t> temp;
      unsigned char tempbuf[140] = {0};
      int get_num = recv(sockfd, tempbuf, 140, MSG_DONTWAIT);
      for (int i = 0; i < get_num; i++)
        temp.push_back(tempbuf[i]);
      uint8_t count = temp.size() / 14;
      uint8_t remain = temp.size() % 14;
      if (count >= 1 && remain == 0)
      {
        for (int i = 0; i < (count - 1) * 14; i++)
        {
          temp.erase(temp.begin());
        }
        for (int i = 0; i < 14; i++)
        {
          buf[i] = temp.at(0);
          temp.erase(temp.begin());
        }
        readtempdata.DatafromStream(buf, 14);
        getframe = true;
      }
    }

    if (getframe)
    {
      if (chacke_data(buf))
      {
        // ROS_INFO("Read: %X %X %X %X %X %X %X %X %X %X %X %X %X %X", buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7], buf[8], buf[9], buf[10], buf[11], buf[12], buf[13]);
        ret = true;
        break;
      }
    }
  } while (iter++ < waitconunt);

  // if (iter >= waitconunt)
  //   ROS_ERROR_STREAM("Read Overtime you can increase 'WaitDataTime' in launch file ");

  return ret;
}
