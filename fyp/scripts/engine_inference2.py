from ctypes import sizeof
import tensorrt as trt
import pycuda.driver as cuda
from util.util import tensor2labelim, tensor2confidencemap
#import pycuda.driver as cuda2
import pycuda.autoinit
import numpy as np
import cv2
import torch
import time
from torchvision import transforms

__image_transform = transforms.Compose([
            transforms.ToTensor()])


imgpath = '/home/simon/catkin_ws/src/fyp/scripts/examples/rgb.png'
image = cv2.imread(imgpath)
print(image.shape)
image = cv2.resize(image,(1248,384))
print(image.shape)
image = torch.tensor(image.astype(np.float32) / 1000)
image = image.unsqueeze(dim=0)
image = image.transpose(1, 3).transpose(2,3).numpy()
print(image.shape)
imgpathdepth = '/home/simon/catkin_ws/src/fyp/scripts/examples/1.png'
imagedepth = cv2.imread(imgpathdepth,cv2.IMREAD_ANYDEPTH)
imagedepth = cv2.resize(imagedepth,(1248,384))
imagedepth = torch.tensor(imagedepth.astype(np.float32) / 65536)
imagedepth = imagedepth.unsqueeze(dim=0).unsqueeze(dim=0)
imagedepth = imagedepth.numpy()
print(imagedepth.shape)
#rgb_image = __image_transform(image.astype(np.float32) / 1000)
#depth_image = __image_transform(imagedepth.astype(np.float32) / 65535)

def load_engine(engine_path):
    #TRT_LOGGER = trt.Logger(trt.Logger.WARNING)  # INFO
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())
 
path ='/home/simon/Desktop/111.engine'
#这里不以某个具体模型做为推断例子.
 
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

# preprocess input data
rgb_input = np.array(image, dtype=np.float32, order='C')
depth_input = np.array(imagedepth, dtype=np.float32, order='C')
print(rgb_input.shape)
print(depth_input.shape)
cuda.memcpy_htod_async(device_input_rgb, rgb_input, stream)
cuda.memcpy_htod_async(device_input_depth, depth_input, stream)
time_start = time.time()
context.execute_async(bindings=[int(device_input_rgb),int(device_input_depth), int(device_output)], stream_handle=stream.handle)

cuda.memcpy_dtoh_async(host_output, device_output, stream)
# for i in range(0,958464):
#     print(host_output[i,])
print(host_output.shape)
stream.synchronize()
time_stop = time.time()
print(time_stop-time_start)
output_data = np.array(host_output.reshape(2,384,1248),dtype = np.uint8)
# for i in range(0,1248):
#     for j in range(0,384):
#         print(output_data[i,j])
output_data = transforms.ToTensor()(output_data).unsqueeze(dim=0).transpose(1,2).transpose(2,3)
print(output_data.shape)
palet_file = '/home/simon/catkin_ws/src/fyp/scripts/datasets/palette.txt'
impalette = list(np.genfromtxt(palet_file, dtype=np.uint8).reshape(3 * 256))
pred_img = tensor2labelim(output_data, impalette)
pred_img = cv2.resize(pred_img,(1280,720))
#output_data = cv2.resize(output_data,(1280,720))

print(type(output_data))
print(output_data.shape)
cv2.imshow('result',pred_img)
cv2.waitKey(0)
