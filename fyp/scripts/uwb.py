#!/usr/bin/env python3

import rospy
import serial
import os
from fyp.msg import Uwb_info
#from geometry_msgs.msg import Vector3

uwb_data = Uwb_info()
os.system('echo %s | sudo -S %s' % ('20001007', 'chmod 777 /dev/ttyUSB0'))
pub = rospy.Publisher('uwbArray', Uwb_info, queue_size = 5)
rospy.init_node('UWBPublisher', anonymous = False)

ser = serial.Serial("/dev/ttyUSB0",115200,timeout=2)

while not rospy.is_shutdown():
    line = ser.readline()
    line = line.decode()
    if line.startswith('mc'):
        print(line)
        uwb_data.uwb_info[0] = int(line[3:5])
        uwb_data.uwb_info[1] = int(line[6:14],16) 
        uwb_data.uwb_info[2] = int(line[15:23],16) 
        uwb_data.uwb_info[3] = int(line[24:32],16) 
        uwb_data.uwb_info[4] = int(line[33:41],16) 
        print(uwb_data.uwb_info)
        pub.publish(uwb_data)






    