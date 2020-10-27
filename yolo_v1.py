import numpy as np
import os
import sys

import torch
import resnet
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
import torchvision.models as models

class ResNet50(nn.Module):  # 1/32
    def __init__(self):
        super(ResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=True)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        return x

class Yolo_v1(nn.Module):
    def __init__(self):
        super(Yolo_v1, self).__init__()
        print("x........")
        
        self.resnet = resnet.resnet50()
        print("x........")
        
        self.detect = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048,  13*13*5),
            )
        print("x........")
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
        
                    
    def forward(self, x):
            
            
            #print("x",x.shape)
            x = self.resnet(x)
            #print("x",x.shape)
            x = x.view(x.size(0), -1)
            
            x_out = self.detect(x)
            b = x_out.size(0)
            
            x_out = x_out.view(b,  13, 13,5)
            
            return x_out

class Yolo_v1_2(nn.Module):
    def __init__(self):
        super(Yolo_v1_2, self).__init__()
        print("x........")
        
        self.resnet = ResNet50()
        print("x........")
        
        self.detect = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048,  13*13*5),
            )
        print("x........")
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        
#         for m in self.detect.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             if isinstance(m, nn.Linear):
#                 nn.init.constant_(m.weight, 0)
#                 nn.init.constant_(m.bias, 0)
        
        
                    
    def forward(self, x):
            
            
            #print("x",x.shape)
            x = self.resnet(x)
            x = self.avgpool(x)
            #print("x",x.shape)
            x = x.view(x.size(0), -1)
            
            x_out = self.detect(x)
            b = x_out.size(0)
            
            x_out = x_out.view(b, 13, 13, 5)
            
            return x_out
"""            
net_det = Yolo_v1_2()
x  = torch.randn(1,3,224,224)

x_out = net_det(x)
print(x_out.shape)
"""
