#!/usr/bin/env python
#coding:utf-8
import torch
import os
os.environ["CUDA_VISIBILE_DEVICES"] = "1"
#import horovod.torch as hvd

#Initialize Horovod
#hvd.init()
#print(hvd.size(), hvd.rank(), hvd.local_rank())
# Pin GPU to be used to process local rank (one GPU per process)
#torch.cuda.set_device(hvd.local_rank())

import sys
import os
import time
import shutil
import torch.nn as nn
import torch.optim
import torchvision.models
import torch
#from input import img_parts
#from tensorboardX import SummaryWriter


def conv33(in_planes, out_planes, stride=1,groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False )

def conv11(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False )

class basic_block(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(basic_block, self).__init__()

        norm_layer = nn.BatchNorm2d   
        self.conv1 = conv33(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = conv33(inplanes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        
        
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
            
        return out
    
class Bottleneck(nn.Module):
    
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64. )) * groups
        #print("width is ",width)
        
        self.conv1 = conv11(inplanes, width)
        self.bn1 = norm_layer(width)
        
        self.conv2 = conv33(width, width, stride,groups,dilation)
        self.bn2 = norm_layer(width)
        
        self.conv3 = conv11(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x
        #print("identity.size() is : ", identity.size())
        
        #print("x",x.size())
        out = self.conv1(x)
        #print("out",out.size())
        out = self.bn1(out)
        out = self.relu(out)
        #print(out.size(),identity.size())
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
            
        #print(out.size(),identity.size())
        out += identity
        
        out = self.relu(out)

        return out
    
    
    
class ResNet(nn.Module):

    def __init__(self, block, layers, num_clsaaes = 1000, zero_init_residual=False, 
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                norm_layer=None):
        
        
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        
        #global replace_stride_with_dilation
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
            
        #if len(replace_stride_with_dilation) !=3:
        
        self.groups = groups
        self.base_width = width_per_group
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0] )
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1] )
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2] )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        num_classes = 15
        self.fc = nn.Linear(512* block.expansion, num_classes)
        
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        
        for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
            norm_layer = self._norm_layer
            downsample = None
            previous_dilation = self.dilation
            
            if dilate:
                self.dilation *= stride
                stride = 1
            if stride!=1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv11(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion)
            )
            layers = []
            #print( "frist conv stride ",stride)
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups, 
                                self.base_width, previous_dilation, norm_layer))

            self.inplanes = planes * block.expansion
            
            
            for _ in range(1, blocks):
                
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
                
                
            return nn.Sequential(*layers)
        
    def _forward_impl(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            #print("x shape ",x.size())

            x = self.layer1(x)
            #print("x shape after2_1",x.size())
            x = self.layer2(x)
            #print("x shape after3_1",x.size())
            x = self.layer3(x)
            #print("x shape after4_1",x.size())
            x = self.layer4(x)
            #print("x shape after5_1",x.size())

            x = self.avgpool(x)
            #x = torch.flatten(x, 1)
            #x = self.fc(x)
            
            return x
        
    def forward(self, x):
            return self._forward_impl(x)
        
        
        
def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)
            
        
        
          
            
            
"""            
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 只能单GPU运行
#net = LeNet().to(device)            
net = resnet50().to(device)  

# Define loss (Cross-Entropy)
import torch.optim as optim
import torchvision.transforms as transforms

criterion = nn.CrossEntropyLoss()
# SGD with momentum
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

transform = transforms.Compose(
     [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
     )        
        
        
        
               
trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        
        
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels =  data[0].to(device), data[1].to(device)
        #print("inputs", inputs.size())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if i % 20 == 19:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1,  running_loss / 2000))
            running_loss = 0.0

print('Finished Training')   
        
"""   
        
        
        
