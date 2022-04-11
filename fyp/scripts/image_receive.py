#!/usr/bin/env python 
# coding=utf-8
from sys import _current_frames
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64MultiArray
pub = rospy.Publisher('speed_info', Vector3, queue_size = 5)
current_frame = np.zeros((720,1280))
ultraDistance = [0.0,0.0,0.0,0.0]
image_x = 0.0
image_y = 0.0
#区域划分参数
CENTRAL_LEFT = 0.125
CENTRAL_RIGHT = 1 - CENTRAL_LEFT
LEFTSIDE = 0.4
RIGHTSIDE = 1 - LEFTSIDE

POSITIVE_PARTIAL = 0.3

def callbackUltra(ultra):
    global ultraDistance
    
    ultraDistance[0] = ultra.data[0]
    ultraDistance[1] = ultra.data[1]
    ultraDistance[2] = ultra.data[2]
    ultraDistance[3] = ultra.data[3]

def callback(data):
    global current_frame
    # Used to convert between ROS and OpenCV images
    br = CvBridge()

    # Output debugging information to the terminal
    rospy.loginfo("receiving video frame")
    # Convert ROS Image message to OpenCV image
    current_frame = br.imgmsg_to_cv2(data)
    print(current_frame[360,720])

    #cv2.imshow("camera", current_frame)
    #print(current_frame[680, 630:650 ])

    #cv2.waitKey(1)
    image_processor()

def image_processor():
    #weighted_counter_array = np.zeros(1280)
    all_counter = 0
    weighted_counter = 0
    weighted_sum = 0
    global current_frame
    central_counter = 0
    turning = 0
    publish_data = Vector3()
    # for i in range(360,720):
    #     for j in range(int(1280*CENTRAL_LEFT),int(1280*CENTRAL_RIGHT)):
    #         if all(current_frame[i,j] == [0,128,0]):
    #             central_counter = central_counter + 1
    grey_img = cv2.cvtColor(current_frame,cv2.COLOR_BGR2GRAY)
    nonzero = cv2.countNonZero(grey_img[360:720, int(1280*CENTRAL_LEFT):int(1280*CENTRAL_RIGHT)])
    
    if nonzero >= int(1280*(1-2*CENTRAL_LEFT)*360*POSITIVE_PARTIAL):
        # for i in range(0,1280):
        #     for j in range(360,720):
        #         if all(current_frame[j,i] == [0,128,0]):
        #             weighted_counter= weighted_counter +1
        #             all_counter = all_counter + 1
        #     weighted_sum = weighted_sum + weighted_counter*(i+1)
        #     weighted_counter = 0
        for i in range (0,1280):
            nonzerocol = cv2.countNonZero(grey_img[360:720,i])
            weighted_sum = weighted_sum + nonzerocol*(i+1)
            all_counter = all_counter + nonzerocol
        average = weighted_sum / all_counter
        average_percent = float((average - 640.0) / 1280.0)
        # if average_percent > 0:
        #     percentageReverse = 0.035 / (0.1 + average_percent) - 0.1
        # elif average_percent < 0:
        #     percentageReverse = -(0.035 / (0.1 - average_percent)) + 0.1
        speed_adjust = float(float(nonzero) / (1280.0*(1.0-2.0*CENTRAL_LEFT)*360.0))
        print("average is %f" %average)
        print("average percent is %f" %average_percent)
        #print("reverse average percent is %f" %percentageReverse)
        print("speed adjust is %f" %speed_adjust)
        publish_data.x = 15+average_percent*100
        publish_data.y = 15-average_percent*100
        publish_data.z = 0
        pub.publish(publish_data)
    else:
        nonzeroleft = cv2.countNonZero(grey_img[360:720, 0:int(1280*LEFTSIDE)])
        nonzeroright = cv2.countNonZero(grey_img[360:720, int(1280*RIGHTSIDE):1280])
        if nonzeroleft <= 360*1280*LEFTSIDE*(POSITIVE_PARTIAL-0.05) and nonzeroright <= 360*1280*LEFTSIDE*(POSITIVE_PARTIAL-0.05):
            publish_data.x = 0
            publish_data.y = 0
            publish_data.z = 0
            pub.publish(publish_data)
            print('too close')
        elif nonzeroleft > 360*1280*LEFTSIDE*(POSITIVE_PARTIAL-0.05) and nonzeroright <= 360*1280*LEFTSIDE*(POSITIVE_PARTIAL-0.05):
            publish_data.x = -5
            publish_data.y = 25
            publish_data.z = 0
            pub.publish(publish_data)
            print('LEFT!!')
        elif nonzeroleft <= 360*1280*LEFTSIDE*(POSITIVE_PARTIAL-0.05) and nonzeroright > 360*1280*LEFTSIDE*(POSITIVE_PARTIAL-0.05):
            publish_data.x = 25
            publish_data.y = -5
            publish_data.z = 0
            pub.publish(publish_data)
            print('RIGHT!!')
        else:
            publish_data.x = 25 if nonzeroleft < nonzeroright else -5
            publish_data.y = -5 if nonzeroleft < nonzeroright else 25
            publish_data.z = 0
            pub.publish(publish_data)
            print('%s' % ('RIGHT!!' if nonzeroleft < nonzeroright else 'LEFT!!' ))

        
        

def listener():
    rospy.init_node("video_sub_py", anonymous=True)
    rospy.Subscriber("segMask", Image, callback)
    rospy.Subscriber("UltraDistanceFront", Float64MultiArray, callbackUltra)
    #rospy.Timer(rospy.Duration(0.1), image_processor, False)
    rospy.spin()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    listener()