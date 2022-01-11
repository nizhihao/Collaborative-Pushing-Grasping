#!/bin/bash

echo "remap serial port(ttyACMx) to DH_hand"
echo "DH_hand connection as /dev/DH_hand , check it using the command : ls -l /dev|grep ttyACM"
echo "start copy DH_hand.rules to  /etc/udev/rules.d/"
echo "`rospack find dh_hand_driver`/scripts/DH_hand.rules"
sudo cp `rospack find dh_hand_driver`/scripts/DH_hand.rules  /etc/udev/rules.d
echo " "
echo "Restarting udev"
echo ""
sudo service udev reload
sudo service udev restart
echo "finish "
echo "If you are already insert the Hand usb, please unplug it and plug again"
