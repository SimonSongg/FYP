import tensorrt as trt
import pycuda.driver as cuda
#import pycuda.driver as cuda2
import pycuda.autoinit
import numpy as np
import cv2
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
context.active_optimization_profile = 0
 
#2. 读取数据，数据处理为可以和网络结构输入对应起来的的shape，数据可增加预处理
imgpath = '/home/simon/catkin_ws/src/fyp/scripts/examples/rgb.png'
image = cv2.imread(imgpath)
image = np.expand_dims(image, 0)  # Add batch dimension.  

imgpathdepth = '/home/simon/catkin_ws/src/fyp/scripts/examples/1.png'
imagedepth = cv2.imread(imgpathdepth,cv2.IMREAD_ANYDEPTH)
imagedepth = np.expand_dims(imagedepth, 0)  # Add batch dimension.  
 
 
#3.分配内存空间，并进行数据cpu到gpu的拷贝
#动态尺寸，每次都要set一下模型输入的shape，0代表的就是输入，输出根据具体的网络结构而定，可以是0,1,2,3...其中的某个头。
context.set_binding_shape(0, image.shape)
d_input = cuda.mem_alloc(image.nbytes)  #分配输入的内存。
 
 
output_shape = context.get_binding_shape(1) 
buffer = np.empty(output_shape, dtype=np.float32)
d_output = cuda.mem_alloc(buffer.nbytes)    #分配输出内存。
cuda.memcpy_htod(d_input,image)
bindings = [d_input ,d_output]
 
#4.进行推理，并将结果从gpu拷贝到cpu。
context.execute_v2(bindings)  #可异步和同步
cuda.memcpy_dtoh(buffer,d_output)  
output = buffer.reshape(output_shape)
 
#5.对推理结果进行后处理。这里只是举了一个简单例子，可以结合官方静态的yolov3案例完善。

