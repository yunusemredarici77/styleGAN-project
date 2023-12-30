from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


def cuda_default(): 
  if torch.cuda.is_available():
    return torch.device('cuda')
  else:
    return torch.device('cpu')
def to_device(ModelData, device):
  if isinstance(ModelData, (list,tuple)):
    return [to_device(i, device) for i in ModelData]
  return ModelData.to(device, non_blocking=True)
class DeviceDataLoader():
  def __init__(self, dataloader, CDevice):
    self.dataloader = dataloader
    self.CDevice = CDevice
  def __iter__(self):
    for b in self.dataloader:
      yield to_device(b, self.CDevice)
  def __len__(self):
    return len(self.dataloader)

initial = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
def d_normalization(tensorOfimage):
  return tensorOfimage * initial[1][0] + initial[0][0]
def show_images(Img, numMax=64):
  figure, xsubplott = plt.subplots(figsize=(10, 10))
  xsubplott.set_xticks([]); xsubplott.set_yticks([])
  xsubplott.imshow(make_grid(d_normalization(Img.detach()[:numMax]), nrow=8).permute(1, 2, 0))
