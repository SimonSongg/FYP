from options.test_options import TestOptions
from models import create_model
from util.util import tensor2labelim, tensor2confidencemap
from models.sne_model import SNE
import torchvision.transforms as transforms
import torch
import numpy as np
import pyrealsense2.pyrealsense2 as rs
import cv2
import os
import time
from torch import nn
import torch.onnx


class dataset():
    def __init__(self):
        self.num_labels = 2


if __name__ == '__main__':

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

    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 5)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 5)

    # Start streaming
    pipeline.start(config)

    ##### Setup for Model #####

    opt = TestOptions().parse()
    opt.num_threads = 1
    opt.batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.isTrain = False

    example_dataset = dataset()
    model = create_model(opt, example_dataset)
    model.setup(opt)
    model.eval()

    ##### Processing #####
    if_need_initialize = True

    try:
        time_start = time.time()
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            exit()
    finally:
        # Stop streaming
        pipeline.stop()

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    rgb_image = np.asanyarray(color_frame.get_data())

    # if you want to use your own data, please modify rgb_image, depth_image, camParam and use_size correspondingly.
    # rgb_image = cv2.cvtColor(cv2.imread(os.path.join('examples', 'rgb.png')), cv2.COLOR_BGR2RGB)
    # depth_image = cv2.imread(os.path.join('examples', '1.png'), cv2.IMREAD_ANYDEPTH)
    oriHeight, oriWidth, _ = rgb_image.shape
    oriSize = (oriWidth, oriHeight)

    # resize image to enable sizes divide 32
    use_size = (1248, 384)
    rgb_image = cv2.resize(rgb_image, use_size)
    rgb_image = rgb_image.astype(np.float32) / 255

    # compute normal using SNE
    sne_model = SNE()
    camParam = torch.tensor([[1361.400024, 0.000000e+00, 955.198975],
                             [0.000000e+00, 1361.399902, 511.039001],
                             [0.000000e+00, 0.000000e+00, 1.000000e+00]],
                            dtype=torch.float32)  # camera parameters

    ## TODO Pure GPU
    normal = sne_model(torch.tensor(depth_image.astype(np.float32) / 1000), camParam)
    normal_image = normal.cpu().numpy()
    normal_image = np.transpose(normal_image, [1, 2, 0])
    # cv2.imwrite(os.path.join('examples', 'normal.png'),
    # cv2.cvtColor(255 * (1 + normal_image) / 2, cv2.COLOR_RGB2BGR))
    normal_image = cv2.resize(normal_image, use_size)

    rgb_image = transforms.ToTensor()(rgb_image).unsqueeze(dim=0).cuda()
    normal_image = transforms.ToTensor()(normal_image).unsqueeze(dim=0).cuda()

    with torch.no_grad():
        pred = model.netRoadSeg(rgb_image, normal_image)
        # Export the model
        torch.onnx.export(model.netRoadSeg.module,  # model being run
                          (rgb_image, normal_image),  # model input (or a tuple for multiple inputs)
                          "roadseg.onnx",
                          # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=11,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['rgb_image', "normal_image"],  # the model's input names
                          output_names=['output'],  # the model's output names
                          )
