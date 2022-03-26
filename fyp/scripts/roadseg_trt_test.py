#!/usr/bin/env python3

from options.test_options import TestOptions
from models import create_model
from util.util import tensor2labelim, tensor2confidencemap
from models.sne_model import SNE
from torchvision import transforms
import torch
import numpy as np
import pyrealsense2 as rs
import cv2
import os
import time
import torch.nn.functional as F
import sys
from torch2trt import TRTModule
import pycuda.autoinit
import tensorrt as trt
import pycuda.driver as cuda

#Define the size of the published image
IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image
rospy.init_node("segPublisher", anonymous=True)
pub = rospy.Publisher('segMask', Image, queue_size = 5)

def load_engine(engine_path):
    #TRT_LOGGER = trt.Logger(trt.Logger.WARNING)  # INFO
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())
 
path ='/home/simon/Desktop/111.engine'

def publish_image(imgdata):
    image_temp=Image()
    header = Header(stamp=rospy.Time.now())
    header.frame_id = 'map'
    image_temp.height=IMAGE_HEIGHT
    image_temp.width=IMAGE_WIDTH
    image_temp.encoding='rgb8'
    image_temp.data=np.array(imgdata).tostring()
    #print(imgdata)
    #image_temp.is_bigendian=True
    image_temp.header=header
    image_temp.step=1280*3
    pub.publish(image_temp)



class dataset():
    def __init__(self):
        self.num_labels = 2


if __name__ == '__main__':
    palet_file = '/home/simon/catkin_ws/src/fyp/scripts/datasets/palette.txt'
    impalette = list(np.genfromtxt(palet_file, dtype=np.uint8).reshape(3 * 256))
    sys.path.extend(['/home/simon/Desktop/ZeyuWANG/SNE-RoadSeg', '/home/simon/Desktop/ZeyuWANG/SNE-RoadSeg'])
    ##### Setup for IntelRealSense #####
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    depth_sensor = device.query_sensors()[0]
    laser_pwr = depth_sensor.get_option(rs.option.laser_power)
    print(laser_pwr)
    depth_sensor.set_option(rs.option.laser_power, 360)

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    # 1. 建立模型，构建上下文管理器
    engine = load_engine(path)
    context = engine.create_execution_context()

    # get sizes of input and output and allocate memory required for input data and for output data
    for binding in engine:
        if engine.binding_is_input(binding):
            if engine.get_binding_shape(binding)[1] == 3:

                input_shape_rgb = engine.get_binding_shape(binding)
                input_size_rgb = trt.volume(input_shape_rgb) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
                device_input_rgb = cuda.mem_alloc(input_size_rgb)
                print(input_shape_rgb)
            else:
                input_shape_depth = engine.get_binding_shape(binding)
                input_size_depth = trt.volume(input_shape_depth) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
                device_input_depth = cuda.mem_alloc(input_size_depth)
        else:  # and one output
            output_shape = engine.get_binding_shape(binding)
            # create page-locked memory buffers (i.e. won't be swapped to disk)
            host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
            device_output = cuda.mem_alloc(host_output.nbytes)

    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()

    # compute normal using SNE
    # sne_model_gpu = SNE("cuda")
    # camParam_gpu = torch.tensor([[1361.400024, 0.000000e+00, 955.198975],
    #                              [0.000000e+00, 1361.399902, 511.039001],
    #                              [0.000000e+00, 0.000000e+00, 1.000000e+00]],
    #                             dtype=torch.float32).cuda()  # camera parameters

    ##### Processing #####

    try:
        while not rospy.is_shutdown():
            time_start = time.time()
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            rgb_image = np.asanyarray(color_frame.get_data())

            image = cv2.resize(rgb_image,(1248,384))
            image = torch.tensor(image.astype(np.float32) / 1000)
            image = image.unsqueeze(dim=0)
            image = image.transpose(1, 3).transpose(2,3).numpy()

            imagedepth = cv2.resize(depth_image,(1248,384))
            imagedepth = torch.tensor(imagedepth.astype(np.float32) / 65536)
            imagedepth = imagedepth.unsqueeze(dim=0).unsqueeze(dim=0)
            imagedepth = imagedepth.numpy()
            
            rgb_input = np.array(image, dtype=np.float32, order='C')
            depth_input = np.array(imagedepth, dtype=np.float32, order='C')
            
            cuda.memcpy_htod_async(device_input_rgb, rgb_input, stream)
            cuda.memcpy_htod_async(device_input_depth, depth_input, stream)
            #### prediction
            context.execute_async(bindings=[int(device_input_rgb),int(device_input_depth), int(device_output)], stream_handle=stream.handle)

            cuda.memcpy_dtoh_async(host_output, device_output, stream)
            stream.synchronize()
            output_data = np.array(host_output.reshape(2,384,1248),dtype = np.uint8)
            output_data = transforms.ToTensor()(output_data).unsqueeze(dim=0).transpose(1,2).transpose(2,3)
            
            
            pred_img = tensor2labelim(output_data, impalette)
            pred_img = cv2.resize(pred_img,(1280,720))
            cv2.imshow('result',pred_img)

            publish_image(pred_img)
            cv2.waitKey(1)
            
    finally:
        # Stop streaming
        pipeline.stop()
