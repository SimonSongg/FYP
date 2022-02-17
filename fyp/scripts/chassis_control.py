#!/usr/bin/env python3

import rospy
import serial
import os
from geometry_msgs.msg import Vector3

os.system('echo %s | sudo -S %s' % ('20001007', 'chmod 777 /dev/ttyTHS0'))

pub = rospy.Publisher('motor', Vector3, queue_size=5)
ser = serial.Serial("/dev/ttyTHS0", 57600, timeout=2)

motorL = 10.0
motorR = 0.0


def callbackSpeed(data):
    global motorL
    global motorR
    motorL = data.x
    motorR = data.y
    # rospy.loginfo


def motorControl(event):
    

    #print(ser)
    
        message="#Ba%s%s%s%s%03d,%03d,%03d,%03d" % (('r' if motorL >= 0 else 'f'), 
          ('r' if motorL >= 0 else 'f'), 
          ('r' if motorR >= 0 else 'f'), 
          ('r' if motorR >= 0 else 'f'), 
          abs(motorL),abs(motorL),abs(motorR), abs(motorR))
        msg_encode = message.encode('ascii')
        print(msg_encode.hex())
        ser.write(msg_encode)
        print(message)
        
        
if __name__ == "__main__":
     rospy.init_node('chassis_motion')
     rospy.Subscriber('speed_info', Vector3, callbackSpeed)
     rospy.Timer(rospy.Duration(0.1), motorControl, False)
     rospy.spin()