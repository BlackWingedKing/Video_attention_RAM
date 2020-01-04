import torch
from torchvision import models
from pprint import pprint
import torch.nn as nn

resnet = models.wide_resnet50_2(pretrained=True)

# model for training

y = torch.randn(10,3,112,112)        
print(y.shape)
a = resnet_base()
print(a(y).shape)