#ifndef DH_DATASTREAM_H
#define DH_DATASTREAM_H
#include <stdint.h>

namespace DH_Robotics{

class DH_DataStream
{
public:
    //Construct a data stream
    DH_DataStream();
    ~DH_DataStream();
    //set dataStream to defalut
    void DataStream_clear();

    /**
     * @brief       : transfer dataStream to uint8_t array
     * 
     * @param buf   : Source array address
     * @param len   : Source array length
     * @return int  : Returns 0 on success; otherwise returns -1. 
     */
    int DatatoStream(uint8_t *buf,uint8_t len);

    /**
     * @brief       :transfer uint8_t array to dataStream
     * 
     * @param buf   : Destination array address
     * @param len   : Destination array length
     * @return int  : Returns 0 on success; otherwise returns -1. 
     */
    int DatafromStream(uint8_t *buf,uint8_t len);

public:
    uint8_t frame_head[4];  //Frame head ,do not modify (defalut : 0xFF 0xFE 0xFD 0xFC ) 
    uint8_t DeviceID;       //Device ID   (defalut : 0x01 )
    uint8_t Register[2];    //register and subRegister (defalur : 0x00 0x00 )
    uint8_t option;         //read or write option
    uint8_t reserve;        //reserve byte
    uint8_t data[4];        //frame data （defalut : 0x00 0x00 0x00 0x00 ） （Little_endian）
    uint8_t frame_end;      //frame end ,do not modify (defalut : 0xFB)
    uint8_t size;           //save dataSteam byte count

};

}
#endif // DH_DATASTREAM_H
