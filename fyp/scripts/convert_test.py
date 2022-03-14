import torch
from torch2trt import torch2trt
from torchvision.models.resnet import resnet18

# create some regular pytorch model...
model = resnet18().eval().cuda()

# create example data
x = torch.ones((1, 3, 224, 224)).cuda()

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x])
