#!/usr/bin/env python 

from sys import _current_frames
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
from geometry_msgs.msg import Vector3
pub = rospy.Publisher('speed_info', Vector3, queue_size = 5)
current_frame = np.zeros((720,1280))

#区域划分参数
CENTRAL_LEFT = 0.125
CENTRAL_RIGHT = 1 - CENTRAL_LEFT
LEFTSIDE = 0.4
RIGHTSIDE = 1 - LEFTSIDE

POSITIVE_PARTIAL = 0.4

def callback(data):
    global current_frame
    # Used to convert between ROS and OpenCV images
    br = CvBridge()

    # Output debugging information to the terminal
    rospy.loginfo("receiving video frame")
    # Convert ROS Image message to OpenCV image
    current_frame = br.imgmsg_to_cv2(data)

    cv2.imshow("camera", current_frame)
    #print(current_frame[680, 630:650 ])

    cv2.waitKey(1)
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
    for i in range[360,720]:
        for j in range[1280*CENTRAL_LEFT,1280*CENTRAL_RIGHT]:
            if current_frame[i,j] == [0,128,0]:
                central_counter = central_counter + 1
    
    if central_counter >= 1280*(1-2*CENTRAL_LEFT)*360*POSITIVE_PARTIAL:
        for i in range[0,1280]:
            for j in range[360,720]:
                if current_frame[j,i] == [0,128,0]:
                    weighted_counter= weighted_counter +1
                    all_counter = all_counter + 1
            weighted_sum = weighted_sum + weighted_counter*(i+1)
            weighted_counter = 0
        average = weighted_sum / all_counter
        average_percent = (average - 640) / 1280
        speed_adjust = central_counter / (1280*(1-2*CENTRAL_LEFT)*360)
        print("average is %f",average)
        print("average percent is %f",average_percent)
        print("speed adjust is %f",speed_adjust)
        publish_data.data = [50+average_percent*25,50-average_percent*25,0]
        pub.Publish(publish_data)
    else:
        
        publish_data.data=[0,0,0]
        pub.Publish(publish_data)
        

def listener():
    rospy.init_node("video_sub_py", anonymous=True)
    rospy.Subscriber("segMask", Image, callback)
    #rospy.Timer(rospy.Duration(0.1), image_processor, False)
    rospy.spin()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    listener()