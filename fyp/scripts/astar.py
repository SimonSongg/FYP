#!/usr/bin/env python3

from logging import PlaceHolder
import rospy
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64MultiArray

import MemapAStar

pub = rospy.Publisher('speed_info', Vector3, queue_size = 5)

memap = MemapAStar.Memap(r'pointcloud.xlsx')

pos_x = 0.0
pos_y = 0.0
nowpos_x = 0.0
nowpos_y = 0.0
ultraDistance = [0,0,0,0]

nextPos = [0,0]

def callbackUltra(ultra):
    global ultraDistance
    
    ultraDistance[0] = ultra.data[0]
    ultraDistance[1] = ultra.data[1]
    ultraDistance[2] = ultra.data[2]
    ultraDistance[3] = ultra.data[3]
    
def callbackCurrentPos(pos):
    global nowpos_x
    global nowpos_y
    
    nowpos_x = pos.x
    nowpos_y = pos.y
    
def callbackPosition(data):
    global pos_x
    global pos_y
    
    pos_x = data.x
    pos_y = data.y
    
def steeringControl(event):
    idontknow = 0.0
    
def tempTarget(event):
    global nowpos_x
    global nowpos_y
    global pos_x
    global pos_y
    global nextPos
    
    memap.aStar((nowpos_x,nowpos_y),(pos_x,pos_y))
    nextPos = memap.findRoute()[1]
    


if __name__ == "__main__":
     rospy.init_node('steering')
     rospy.Subscriber('target', Vector3, callbackPosition)
     rospy.Subscriber('uwb_position', Vector3, callbackCurrentPos)
     rospy.Subscriber('UltraDistanceFront', Float64MultiArray, callbackUltra)
     rospy.Timer(rospy.Duration(0.1), steeringControl, False)
     rospy.spin()