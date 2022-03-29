#!/usr/bin/env python3

from logging import PlaceHolder
import rospy
from geometry_msgs.msg import Vector3,Pose2D
from std_msgs.msg import Float64MultiArray
import math
import MemapAStar

pub = rospy.Publisher('speed_info', Vector3, queue_size = 5)

memap = MemapAStar.Memap(r'pointcloud.xlsx')
# 参数
#MAG_YAW_OFFSET = 0.10402162   # 磁北与正北之间的矫正(radius)
PID_P_FACTOR = 200.0 #PID P系数 正数
PID_I_FACTOR = 0.0  #PID I系数 正数
PID_D_FACTOR = 0.0  #PID D系数 正数
FORWARD_SPEED_MAX_PERCENT = 100.0   #前进油门百分比 0-100

DESTINATION_RANGE = 200 #终点符合范围：厘米
tarpos_x = 0.0
tarpos_y = 0.0
nowpos_x = 0.0
nowpos_y = 0.0
ultraDistance = [0,0,0,0]
yaw = 0.0
nextPos = [0,0]
isTargetReached = True  # 是否到达目的地坐标

# def callbackUltra(ultra):
#     global ultraDistance
    
#     ultraDistance[0] = ultra.data[0]
#     ultraDistance[1] = ultra.data[1]
#     ultraDistance[2] = ultra.data[2]
#     ultraDistance[3] = ultra.data[3]
def callbackYaw(data):
    global yaw
    if (data.theta <= math.pi):
        yaw = -data.theta
    else :
        yaw = -data.theta + 2 * math.pi
    #yaw = data.theta - math.pi
def callbackCurrentPos(pos):
    global nowpos_x
    global nowpos_y
    
    nowpos_x = pos.x
    nowpos_y = pos.y
    
def callbackPosition(data):
    global tarpos_x
    global tarpos_y
    
    tarpos_x = data.x
    tarpos_y = data.y
    
def steeringControl(event):
    idontknow = 0.0
    
def tempTarget(event):
    global nowpos_x
    global nowpos_y
    global tarpos_x
    global tarpos_y
    global nextPos
    
    memap.aStar((nowpos_x,nowpos_y),(tarpos_x,tarpos_y))
    nextPos = memap.findRoute()[1]
def controlPID(event):
    # 参数

    global PID_P_FACTOR
    global PID_I_FACTOR
    global PID_D_FACTOR
    global FORWARD_SPEED_MAX_PERCENT


    # 订阅的值
    global isTargetReached
    global yaw

    #内部处理的值
    global realYaw
    global targetYaw
    global errorYaw
    global lastErrorYaw
    global integrationErrorYaw
    global differenceErrorYaw

    if not isTargetReached:
        # 打印
        rospy.loginfo("Car is still on the way to next node.")

        # 进行一些计算更新值
        realYaw = yaw + MAG_YAW_OFFSET
        targetYaw = math.atan2(longitude - targetLongitude, targetLatitude - latitude)
        lastErrorYaw = errorYaw
        errorYaw = targetYaw - realYaw
        integrationErrorYaw = errorYaw + integrationErrorYaw
        differenceErrorYaw = errorYaw - lastErrorYaw
        outputTurnByGPS = errorYaw * PID_P_FACTOR + integrationErrorYaw * PID_I_FACTOR + differenceErrorYaw * PID_D_FACTOR
        
        # 打印
        rospy.loginfo("latitude:%f\tlongitude:%f" % (latitude, longitude))
        rospy.loginfo("targetLa:%f\ttargetLo:%f" % (targetLatitude, targetLongitude,))
        rospy.loginfo("realYaw:%f\ttarYaw:%f\terYaw:%f" % (realYaw, targetYaw, errorYaw))
        rospy.loginfo("outputTurnByGPS:%f" % (outputTurnByGPS))

        # 雷达
        radarFactor, radarTurn, isEmStop = controlRadar()

        # 打印
        rospy.loginfo("rdFactor:%f\trdTurn:%f\trdIsEM:%f" % (radarFactor, radarTurn, isEmStop))

        # 输出
        outputX = FORWARD_SPEED_MAX_PERCENT * (1 - radarFactor)
        outputY = (constrain(outputTurnByGPS, -99, 99) * (1 - radarFactor) + radarTurn * radarFactor)
        vector3 = Vector3()
        vector3.x = outputX
        vector3.y = outputY
        if isEmStop:
            vector3.z = 1.0
        else:
            vector3.z = 0.0

        # 打印
        rospy.loginfo("fnOutX:%f\tfnOutY:%f\tfnOutEm:%f" % (vector3.x, vector3.y, False if vector3.z == 0 else True))
        
        # 发布
        pub.publish(vector3)

        # 检测是否到达
        distance = math.sqrt(math.pow(abs(targetLatitude - latitude) * LATITUDE_TO_DISTANCE_FACTOR, 2) + math.pow(abs(targetLongitude - longitude) * LONGITUDE_TO_DISTANCE_FACTOR, 2))

        # 打印
        rospy.loginfo("isTargetReached:%s\tdistanceLeft:%f" % (isTargetReached, distance))

    else:
        rospy.loginfo("Car has reached its destination.")
        vector3 = Vector3()
        vector3.x = 0.0;  vector3.y = 0.0; vector3.z = 0.0
        pub.publish(vector3)

        # 打印
        rospy.loginfo("isTargetReached:%s" % (isTargetReached))


if __name__ == "__main__":
     rospy.init_node('steering')
     rospy.Subscriber('target', Vector3, callbackPosition)
     rospy.Subscriber('uwb_position', Vector3, callbackCurrentPos)
     # rospy.Subscriber('UltraDistanceFront', Float64MultiArray, callbackUltra)
     rospy.Subscriber('mag_pose_2d', Pose2D, callbackYaw)                #订阅包含磁方向的Yaw角数据
     rospy.Timer(rospy.Duration(0.1), steeringControl, False)
     rospy.spin()