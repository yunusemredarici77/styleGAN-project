import torch.nn as nn
import torchvision.transforms as tt

import argparse
import random
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.animation as animation
from IPython.display import HTML
import torch.nn as nn
import torch.nn.functional as F


def single_conv(inputChannel, outputChannel, kernelSize, stride_=2, padding_=1, batchNormalization=False, instanceNormalization=False):

    
    #Add layers
    all_layers = []
    convolution_layer = nn.Conv2d(in_channels=inputChannel, out_channels=outputChannel, kernel_size=kernelSize, stride=stride_, padding=padding_, dilation=1, groups=1, bias=False, padding_mode='zeros', device=None, dtype=None)
    all_layers.append(convolution_layer)
        
        
    #Batch normalization
    if batchNormalization:
        all_layers.append(nn.BatchNorm2d(outputChannel))
    #Instance normalization
    if instanceNormalization:
        all_layers.append(nn.InstanceNorm2d(outputChannel))
    return nn.Sequential(*all_layers)

class DiscriminativeModel(nn.Module):    
    def __init__(self, convolutional_dimention=64):
        super(DiscriminativeModel, self).__init__()

        
        #convolutionalLayers
        self.convolution1 = single_conv(inputChannel=3, outputChannel=convolutional_dimention, kernelSize=4, stride_=2, padding_=1, batchNormalization=False, instanceNormalization=False) # dimention of  [128 x 128 x 64]
        self.convolution2 = single_conv(inputChannel=convolutional_dimention, outputChannel=convolutional_dimention*2, kernelSize=4, stride_=2, padding_=1, batchNormalization=False, instanceNormalization=True) #dimention of  [64 x 64 x 128]
        self.convolution3 = single_conv(inputChannel=convolutional_dimention*2, outputChannel=convolutional_dimention*4, kernelSize=4,stride_=2, padding_=1, batchNormalization=False, instanceNormalization=True) # dimention of  [32 x 32 x 256]
        self.convolution4 = single_conv(inputChannel=convolutional_dimention*4, outputChannel=convolutional_dimention*8, kernelSize=4,stride_=2, padding_=1, batchNormalization=False, instanceNormalization=True) # dimention of  [16 x 16 x 512]
        self.convolution5 = single_conv(inputChannel=convolutional_dimention*8, outputChannel=convolutional_dimention*8, kernelSize=4,stride_=2, padding_=1, batchNormalization=False, instanceNormalization=True) # dimention of  [8 x 8 x 512]
        self.convolution6 = single_conv(convolutional_dimention*8, outputChannel=1, kernelSize=4, stride_=1, padding_=1) # dimention of  [8 x 8 x 1]
    
    def forward(self, a):
        
        #leaky_relu activation function is applied to layers
        output = F.leaky_relu(self.convolution1(a), negative_slope=0.2)
        output = F.leaky_relu(self.convolution2(output), negative_slope=0.2)
        output = F.leaky_relu(self.convolution3(output), negative_slope=0.2)
        output = F.leaky_relu(self.convolution4(output), negative_slope=0.2)

        output = self.convolution6(output)
        return output
    
    
class ResidualBlock(nn.Module):

    def __init__(self, convolutional_dimention):
        super(ResidualBlock, self).__init__()

        self.convolution1 = single_conv(inputChannel=convolutional_dimention, outputChannel=convolutional_dimention, kernelSize=3, stride_=1, padding_=1,batchNormalization=False, instanceNormalization=True)
        self.convolution2 = single_conv(inputChannel=convolutional_dimention, outputChannel=convolutional_dimention, kernelSize=3, stride_=1, padding_=1, batchNormalization=False, instanceNormalization=True)
     
    def forward(self, a):
        out_1 = F.relu(self.convolution1(a))
        out_2 = a + self.convolution2(out_1)
        return out_2

def deconvolution(inputChannel, outputChannel, kernelSize, stride_=2, padding_=1, batchNormalization=False, instanceNormalization=False, dropout_=False, dropoutRatio=0.5):

    

    layers = []
    layers.append(nn.ConvTranspose2d(inputChannel, outputChannel, kernelSize, stride_, padding_, bias=False))
    
    #Apply Batch Normalization 
    if batchNormalization:
        layers.append(nn.BatchNorm2d(outputChannel))
    
    #Apply Instance Normalization 
    if instanceNormalization:
        layers.append(nn.InstanceNorm2d(outputChannel))
    #Apply Dropout
    if dropout_:
        layers.append(nn.Dropout2d(dropoutRatio))
    
    return nn.Sequential(*layers)


class CycleGenerativeModel(nn.Module):   
    def __init__(self, convolution_dimention=64, NumberOfResidualBlock=6):
        super(CycleGenerativeModel, self).__init__()
 
        
        #Encoder_Laayers
        self.convolution1 = single_conv(inputChannel=3, outputChannel=convolution_dimention, kernelSize=4) # dimention of  [128 x 128 x 64]
        self.convolution2 = single_conv(inputChannel=convolution_dimention, outputChannel=convolution_dimention*2, kernelSize=4, instanceNormalization=True)# dimention of  [64 x 64 x 128]
        self.convolution3 = single_conv(inputChannel=convolution_dimention*2, outputChannel=convolution_dimention*4, kernelSize=4, instanceNormalization=True) # dimention of  [32 x 32 x 256]
                #Residual_Blocks (number depends on input parameter)
        residual_layers = []
        for layer in range(NumberOfResidualBlock):
            residual_layers.append(ResidualBlock(convolution_dimention*4))
        self.residual_blocks = nn.Sequential(*residual_layers)
        
        #DecoderLayers
        self.convolution4 = deconvolution(inputChannel=convolution_dimention*4, outputChannel=convolution_dimention*2, kernelSize=4, stride_=2, padding_=1, batchNormalization=False, instanceNormalization=True, dropout_=False, dropoutRatio=0.5)# dimention of  [64 x 64 x 128] 
        self.convolution5 = deconvolution(inputChannel=convolution_dimention*2, outputChannel=convolution_dimention, kernelSize=4, stride_=2, padding_=1, batchNormalization=False, instanceNormalization=True, dropout_=False, dropoutRatio=0.5) # dimention of  [128 x 128 x 64]
        self.convolution6 = deconvolution(inputChannel=convolution_dimention, outputChannel=3, kernelSize=4, stride_=2, padding_=1, batchNormalization=False, instanceNormalization=True, dropout_=False, dropoutRatio=0.5) # dimention of  [256 x 256 x 3]
    def forward(self, a):

        #EncoderForward
        output = F.leaky_relu(self.convolution1(a), negative_slope=0.2)  # dimention of  [128 x 128 x 64]
        output = F.leaky_relu(self.convolution2(output), negative_slope=0.2) # dimention of  [64 x 64 x 128]
        output = F.leaky_relu(self.convolution3(output), negative_slope=0.2) # dimention of  [32 x 32 x 256]
        
        #Residual_Blocks_Forward
        output = self.residual_blocks(output)
        
        #DecoderForward
        output = F.leaky_relu(self.convolution4(output), negative_slope=0.2) # dimention of  [64 x 64 x 128] 
        output = F.leaky_relu(self.convolution5(output), negative_slope=0.2) # dimention of  [128 x 128 x 64]
        output = torch.tanh(self.convolution6(output)) # dimention of  [256 x 256 x 3]
        
        return output



"""
DiscriminativeModel = nn.Sequential(
    # in: 3 x 64 x 64
    nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros', device=None, dtype=None),
    nn.BatchNorm2d(64),
    nn.SELU(inplace=False),
    # out: 64 x 32 x 32
    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1,dilation=1, groups=1, bias=False, padding_mode='zeros', device=None, dtype=None),
    nn.BatchNorm2d(128),
    nn.SELU(inplace=False),
    # out: 128 x 16 x 16
    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1,dilation=1, groups=1, bias=False, padding_mode='zeros', device=None, dtype=None),
    nn.BatchNorm2d(256),
    nn.SELU(inplace=False),
    # out: 256 x 8 x 8
    nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1,dilation=1, groups=1, bias=False, padding_mode='zeros', device=None, dtype=None),
    nn.BatchNorm2d(512),
    nn.SELU(inplace=False),
    # out: 512 x 4 x 4
    nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0,dilation=1, groups=1, bias=False, padding_mode='zeros', device=None, dtype=None),
    # out: 1 x 1 x 1
    nn.Flatten(),
    nn.Sigmoid())
SizeOfLatent = 128
generativeModel = nn.Sequential(
    # in: latent_size x 1 x 1
    nn.ConvTranspose2d(SizeOfLatent, 512, kernel_size=4, stride=1, padding=0, output_padding=0, groups=1, bias=False , padding_mode='zeros', device=None, dtype=None),
    nn.BatchNorm2d(512),
    nn.GELU(approximate='none'),
    # out: 512 x 4 x 4
    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1,output_padding=0, groups=1, bias=False , padding_mode='zeros', device=None, dtype=None),
    nn.BatchNorm2d(256),
    nn.GELU(approximate='none'),
    # out: 256 x 8 x 8
    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0, groups=1, bias=False , padding_mode='zeros', device=None, dtype=None),
    nn.BatchNorm2d(128),
    nn.GELU(approximate='none'),
    # out: 128 x 16 x 16
    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0, groups=1, bias=False , padding_mode='zeros', device=None, dtype=None),
    nn.BatchNorm2d(64),
    nn.GELU(approximate='none'),
    # out: 64 x 32 x 32
    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, output_padding=0, groups=1, bias=False , padding_mode='zeros', device=None, dtype=None),
    nn.Tanh()
    # out: 3 x 64 x 64
    )
"""


