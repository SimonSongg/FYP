#!/usr/bin/env python3

import rospy
import serial
import os
from std_msgs.msg import String
os.system('echo %s | sudo -S %s' % ('20001007', 'chmod 777 /dev/ttyUSB2'))
pub = rospy.Publisher('blemsg', String, queue_size = 5)
rospy.init_node('blePublisher', anonymous = False)
ser = serial.Serial("/dev/ttyUSB2",9600,timeout=0.1)

while not rospy.is_shutdown():
    line = ser.readline()
    line = line.decode()
    print(line)
    if line == 'f':
        print("fuckkkkkk")
        pubdata = String()
        pubdata.data = "fuckkkkk"
        pub.publish(pubdata)


    
