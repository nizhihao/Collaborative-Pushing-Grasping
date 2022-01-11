#include <vector>
#include "dh_hand_driver/hand_driver.h"
DH_Hand_Base::DH_Hand_Base(): velocity_(0),position_1(0),target_reached_1(false)
{
}

DH_Hand_Base::~DH_Hand_Base()
{
}

void DH_Hand_Base::reset()
{
  
}

std::vector<uint8_t> DH_Hand_Base::getStream()
{
  std::vector<uint8_t> temp_data;
  uint8_t buf[14];
  mDatastream.DatatoStream(buf,mDatastream.size);
  for(int i = 0;i<14;i++)
     temp_data.push_back(buf[i]);
  return temp_data;
}

void DH_Hand_Base::SetOpration(int Reg,int data,int Write,int submode)
{
    mDatastream.DataStream_clear();
    mDatastream.Register[0]=Reg &0xFF;
    mDatastream.Register[1]=submode&0xFF;
    mDatastream.option=Write&0xFF;
    mDatastream.data[0]=data&0xFF;
    mDatastream.data[1]=(data>>8)&0xFF;
    mDatastream.data[2]=(data>>16)&0xFF;
    mDatastream.data[3]=(data>>24)&0xFF;
}

void DH_Hand_Base::setInitialize()
{
    //FF FE FD FC 01 08 02 01 00 00 00 00 00 FB 
    SetOpration();
}

void DH_Hand_Base::setMotorPosition(int MotorID ,const int &target_position)
{
    if(MotorID == 1)
        SetOpration(DH_Robotics::R_Posistion_1,target_position,DH_Robotics::Write);
    else if(MotorID == 2)
        SetOpration(DH_Robotics::R_Posistion_2,target_position,DH_Robotics::Write);

}

void DH_Hand_Base::setMotorForce(const int &target_force)
{
    //FF FE FD FC 01 05 02 01 00 5A 00 00 00 FB
    SetOpration(DH_Robotics::R_Force,target_force,DH_Robotics::Write);
}

void DH_Hand_Base::getMotorPosition(const int &MotorID)
{
    //FF FE FD FC 01 06 02 01 00 5A 00 00 00 FB
    if(MotorID == 1)
        SetOpration(DH_Robotics::R_Posistion_1,0,DH_Robotics::Read);
    else if(MotorID == 2)
        SetOpration(DH_Robotics::R_Posistion_2,0,DH_Robotics::Read);
}

void DH_Hand_Base::getMotorForce()
{
    //FF FE FD FC 01 05 02 01 00 5A 00 00 00 FB
    SetOpration(DH_Robotics::R_Force,0,DH_Robotics::Read);
}

void DH_Hand_Base::getFeedback(const int &MotorID)
{
    SetOpration(DH_Robotics::R_Feedback,0,DH_Robotics::Read,MotorID);
}
