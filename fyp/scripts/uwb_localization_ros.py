#!/usr/bin/env python3

import rospy
import os
from geometry_msgs.msg import Vector3
from std_msgs.msg import Int64MultiArray
from fyp.msg import Uwb_info


from ctypes import *

import _ctypes

import ctypes

#Trilateration lib import
CUR_PATH=os.path.dirname(__file__)
dllPath=os.path.join(CUR_PATH,"libtest.so")
pDll=ctypes.cdll.LoadLibrary(dllPath)
result=0

class UWBMsg(Structure):

     _fields_ = [("x", c_double),

                ("y", c_double),

                ("z", c_double)]
location = UWBMsg()

anchorArray=(UWBMsg*3)()

distanceArray=(c_int*3)()
#distanceArray=[0,0,0]


anchorArray[0].x=0

anchorArray[0].y=0

anchorArray[0].z=2



anchorArray[1].x=0.87

anchorArray[1].y=0

anchorArray[1].z=2



anchorArray[2].x=0

anchorArray[2].y=1.97

anchorArray[2].z=2

#ROS publisher init
pub = rospy.Publisher('uwb_position', Vector3, queue_size = 5)
position_data = Vector3()

#distanceArray = [ctypes.c_int(7100),ctypes.c_int(7100),ctypes.c_int(7100)]

def callbackPosition(msgg):
    global distanceArray

    distanceArray[0] = msgg.data[1]
    distanceArray[1] = msgg.data[2]
    distanceArray[2] = msgg.data[3]

def calculatePosition(event):
    global distanceArray
    global location
    global anchorArray
    #print(location.x)

    result=pDll.GetLocation(byref(location),0,anchorArray,distanceArray)
    print(location.x)
    position_data.x = location.x
    print(location.y)
    position_data.y = location.y
    print(location.z)
    position_data.z = location.z
    pub.publish(position_data)



if __name__ == "__main__":
     rospy.init_node('position_publisher', anonymous = False)
     rospy.Subscriber('uwbArray', Int64MultiArray, callbackPosition)
     rospy.Timer(rospy.Duration(0.1), calculatePosition, False)
     rospy.spin()