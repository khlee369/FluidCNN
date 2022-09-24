import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import numpy as np


class BlockConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_rate=0.5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout_rate))
        
    def forward(self, x):
        return self.block(x)
        
class BlockDense(nn.Module):
    def __init__(self, in_size, out_size, dropout_rate=0.5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.BatchNorm1d(out_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate))
        
    def forward(self, x):
        return self.block(x)

class FluidCNN(nn.Module):
    def __init__(self, in_features=12, in_ch=3, out_dim=256*256):
        super().__init__()
        self.flatten_size = in_features*128
        
        self.block1_conv1d = BlockConv1D(in_ch, 32)
        self.block2_conv1d = BlockConv1D(32, 128)
        self.flatten = nn.Flatten()
        self.block3_dense = BlockDense(in_features*128,32)
        self.block4_dense = BlockDense(32,256)
        self.block5_dense = BlockDense(256,512)
        self.last_layer = nn.Linear(512, out_dim)

    def forward(self, x):
        x = self.block1_conv1d(x)
        x = self.block2_conv1d(x)
        x = self.flatten(x)
        x = self.block3_dense(x)
        x = self.block4_dense(x)
        x = self.block5_dense(x)
        x = self.last_layer(x)
        return x