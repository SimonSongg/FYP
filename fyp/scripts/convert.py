import torch
import torch.nn as nn
from models.networks import RoadSeg
import tensorrt as trt
from torch2trt import torch2trt
import torchvision.models as models

pthfile = '/home/simon/catkin_ws/src/fyp/scripts/checkpoints/nosne_test/best_net_RoadSeg.pth'
model = RoadSeg(2,False)
model.load_state_dict(torch.load(pthfile))
model.eval().cuda()
#print(net)
x = torch.ones((1,3,384,1248)).cuda()
y = torch.ones((1,1,384,1248)).cuda()
print("start")
model_trt = torch2trt(model,[x,y])

torch.save(model_trt.state_dict, '111.pth')
