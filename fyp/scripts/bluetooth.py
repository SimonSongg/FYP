#!/usr/bin/env python3

import rospy
import serial
import os
from geometry_msgs.msg import Vector3
os.system('echo %s | sudo -S %s' % ('20001007', 'chmod 777 /dev/ttyUSB0'))
pub = rospy.Publisher('speed_info', Vector3, queue_size = 5)
rospy.init_node('blePublisher', anonymous = False)
ser = serial.Serial("/dev/ttyUSB0",9600,timeout=0.1)

while not rospy.is_shutdown():
    line = ser.readline()
    line = line.decode()
    print(line)
    if line == 's':
        print("stoppppp!!")
        pubdata = Vector3()
        pubdata.data = [0,0,1]
        pub.publish(pubdata)
    elif line == 'w':
        print("goooooo!!")
        pubdata = Vector3()
        pubdata.data = [0,0,2]
        pub.publish(pubdata)


    
