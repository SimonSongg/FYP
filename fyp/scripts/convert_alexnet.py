import torch
import torch.nn as nn
from models.networks import RoadSeg
import tensorrt as trt
from torch2trt import torch2trt
import torchvision.models as models
from torchvision.models.alexnet import alexnet
input_name = ['x','y']
output_name = ['outputimg']
pthfile = '/home/simon/catkin_ws/src/fyp/scripts/checkpoints/nosne_test/best_net_RoadSeg.pth'
model = RoadSeg(2,False)
model.load_state_dict(torch.load(pthfile))
model.eval().cuda()

x = torch.ones((1,3,384,1248)).cuda()
y = torch.ones((1,1,384,1248)).cuda()

outputimg = torch.ones((1,3,384,1248)).cuda()

torch.onnx.export(model, (x,y), '111.onnx', input_names=input_name, output_names=output_name, verbose=True,opset_version=11)