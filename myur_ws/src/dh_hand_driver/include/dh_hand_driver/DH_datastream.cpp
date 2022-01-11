
#include "DH_datastream.h"
#include <stdint.h>
#include <string.h>
DH_Robotics::DH_DataStream::DH_DataStream()
{
    DataStream_clear();
}

DH_Robotics::DH_DataStream::~DH_DataStream()
{

}

void DH_Robotics::DH_DataStream::DataStream_clear()
{
    frame_head[0]=0xFF;
    frame_head[1]=0xFE;
    frame_head[2]=0xFD;
    frame_head[3]=0xFC;
    DeviceID=0x01;
    Register[0]=0x00;
    Register[1]=0x00;
    option=0x00;
    reserve=0x00;
    data[0]=0x00;
    data[1]=0x00;
    data[2]=0x00;
    data[3]=0x00;
    frame_end=0xFB;
    size = (uint8_t)(&size - frame_head);
}

int DH_Robotics::DH_DataStream::DatatoStream(uint8_t *buf,uint8_t len)
{
    if(len!=size)
    {
        return -1;
    }
    memcpy(buf,frame_head,len);
    return 0;

}

int DH_Robotics::DH_DataStream::DatafromStream(uint8_t *buf,uint8_t len)
{
    if(len!=size)
    {
        return -1;
    }
    memcpy(frame_head,buf,len);
    return 0;
}