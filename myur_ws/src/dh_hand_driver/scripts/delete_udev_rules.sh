#!/bin/bash

echo "delete remap tserial port(ttyACMx) to DH_hand"
echo "sudo rm /etc/udev/rules.d/DH_hand.rules"
sudo rm   /etc/udev/rules.d/DH_hand.rules
echo " "
echo "Restarting udev"
echo ""
sudo service udev reload
sudo service udev restart
echo "finish  delete"
