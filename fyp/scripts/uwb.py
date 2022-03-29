#!/usr/bin/env python3

import rospy
import serial
import os
from fyp.msg import Uwb_info
from std_msgs.msg import Int64MultiArray
#from geometry_msgs.msg import Vector3
data_array = [0,0,0,0,0]
uwb_data = Uwb_info()
os.system('echo %s | sudo -S %s' % ('20001007', 'chmod 777 /dev/ttyUSB2'))
pub = rospy.Publisher('uwbArray', Int64MultiArray, queue_size = 5)
rospy.init_node('UWBPublisher', anonymous = False)

ser = serial.Serial("/dev/ttyUSB2",115200,timeout=2)

while not rospy.is_shutdown():
    line = ser.readline()
    line = line.decode()
    if line.startswith('mc'):
        print(line)
        data_array[0] = int(line[3:5])
        data_array[1] = int(line[6:14],16) 
        data_array[2] = int(line[15:23],16) 
        data_array[3] = int(line[24:32],16) 
        data_array[4] = int(line[33:41],16) 
        print(data_array)
        publishdata = Int64MultiArray(data=data_array)
        pub.publish(publishdata)






    