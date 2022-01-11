/**
 * Definition of the commands of the server motors and the stepper motors 
 * auther: Jie Sun
 * email: jie.sun@dh-robotics.com
 * 
 * date: 2018 April 23
*/

// For DH HAND

#ifndef HAND_CONTROLLER_DEFINITION
#define HAND_CONTROLLER_DEFINITION

namespace DH_Robotics
{

enum OPTION
{
    Read = 0x00,
    Write = 0x01
};

enum RegisterType
{
    R_Force = 0x05,
    R_Posistion_1 = 0x06,
    R_Posistion_2 = 0x07,
    R_Initialization = 0x08,
    R_Feedback = 0x0f,
    R_Slowmode = 0x11
};

enum SlowModeType
{
    S_stop = 0x00,
    S_close = 0x01,
    S_open = 0x02,
};

enum FeedbackType
{
    UNDEFINE = 0,
    ARRIVED = 2, 
    CATCHED = 3, 
};

} // namespace DH_Robotics

#endif
