#!/usr/bin/env python

import rospy
import serial
import os
from geometry_msgs.msg import Vector3

os.system('echo %s | sudo -S %s' % ('20001007', 'chmod 777 /dev/ttyTHS0'))

pub = rospy.Publisher('motor', Vector3, queue_size=5)
ser = serial.Serial("/dev/ttyTHS0", 57600, timeout=2)

motorL = 0.0
motorR = 0.0


def callbackSpeed(data):
    global motorL
    global motorR
    motorL = data.x
    motorR = data.y
    # rospy.loginfo


def motorControl(speedL, speedR):
    

    #print(ser)
    while(1):
        message="#Ba%s%s%s%s%03d,%03d,%03d,%03d" % (('r' if speedL >= 0 else 'f'), 
          ('r' if speedL >= 0 else 'f'), 
          ('r' if speedR >= 0 else 'f'), 
          ('r' if speedR >= 0 else 'f'), 
          abs(speedL),abs(speedL),abs(speedR), abs(speedR))
        msg_encode = message.encode('ascii')
        print(msg_encode.hex())
        ser.write(msg_encode)
        print(message)
        
        
if __name__ == "__main__":
     rospy.Subscriber('speed_info', Vector3, callbackSpeed)
     rospy.Timer(rospy.Duration(0.1), motorControl(motorL,motorR), False)