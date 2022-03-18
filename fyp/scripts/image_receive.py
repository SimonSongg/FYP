#!/usr/bin/env python 

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np

def callback(data):
    # Used to convert between ROS and OpenCV images
    br = CvBridge()

    # Output debugging information to the terminal
    rospy.loginfo("receiving video frame")
    # Convert ROS Image message to OpenCV image
    current_frame = br.imgmsg_to_cv2(data)

    cv2.imshow("camera", current_frame)
    print(current_frame[680, 630:650 ])

    cv2.waitKey(1)


def listener():
    rospy.init_node("video_sub_py", anonymous=True)
    rospy.Subscriber("segMask", Image, callback)

    rospy.spin()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    listener()