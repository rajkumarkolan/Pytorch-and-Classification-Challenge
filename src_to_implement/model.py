import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self,input_channels,output_channels,stride,downsample=None):
        super(ResBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels,out_channels=output_channels,kernel_size=3,stride=stride)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(in_channels=output_channels,out_channels=output_channels,kernel_size=3)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self,input):
        residual = input
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)
        if self.downsample is not None:
            residual = self.downsample(input)
        output += residual
        output = self.relu(output)
        return output

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet,self).__init__()
        self.output_channels = 64
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pooling = nn.MaxPool2d(kernel_size=3,stride=2)
        #self.res_block1 = self.make_layer(Resblock,channels_out=64, stride=1)
        self.res_block1 = ResBlock()
        self.res_block2 = self.make_layer(ResBlock,channels_out=128, stride=2)
        self.res_block3 = self.make_layer(ResBlock,channels_out=256, stride=2)
        self.res_block4 = self.make_layer(ResBlock,channels_out=512, stride=2)
        self.globalavgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_features=512,out_features=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self,input_tensor):
        input_tensor = self.conv1(input_tensor)
        input_tensor = self.bn(input_tensor)
        input_tensor = self.relu(input_tensor)
        input_tensor = self.max_pooling(input_tensor)
        input_tensor = self.res_block1(input_tensor)
        input_tensor = self.res_block2(input_tensor)
        input_tensor = self.res_block3(input_tensor)
        input_tensor = self.res_block4(input_tensor)
        input_tensor = self.globalavgpool(input_tensor)
        input_tensor = input_tensor.view(input_tensor.size(0),-1)    # flatten
        input_tensor = self.fc(input_tensor)
        input_tensor = self.sigmoid(input_tensor)
        return input_tensor

    def make_layer(self,ResBlock,channels_out,stride):
        layers = []
        downsample = nn.Sequential()
        layers.append(ResBlock(self.output_channels,channels_out,stride,downsample))
        self.output_channels = channels_out
        return nn.Sequential(*layers)