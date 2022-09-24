import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torch
import torchvision.transforms as T
import cv2

class SampleData(Dataset):
    def __init__(self):
        # time dimension : 3840
        # region dimension(channel) : 3
        # channles comes first than feature in torch data dimesion
        # 
        self.input = torch.ones(3, 3840)
        self.output = torch.ones(256*256, 3840)
        self.time_div = 12
        self.len = self.input.shape[1]-13 # 3840 - 12(hours) - 1(predict hour)

    def __getitem__(self, idx):
        input_x = self.input[:, idx:idx+self.time_div]
        target_y = self.output[:, idx+self.time_div+1]
        return input_x, target_y

    def __len__(self):
        return self.len
