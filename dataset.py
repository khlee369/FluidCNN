import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torch
import torchvision.transforms as T
import cv2
import pandas as pd
import traceback

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

class FluidData(Dataset):
    def __init__(self, dpath):
        # time dimension : 3840
        # region dimension(channel) : 3
        # channles comes first than feature in torch data dimesion
        self.dsize = 256
        self.pd = pd.read_csv(dpath, header=None, sep='\n')
        self.pd = self.pd[0][1:-1] # drop header and last garbage line

        self.time_div = 12
        self.len = len(self.pd) - self.time_div - 1 # 3840 - 12(hours) - 1(predict hour)

    def __getitem__(self, idx):
        idx += 1 # pandas table start with idx==1
        input_x = np.zeros([3,self.time_div], dtype=np.float32)
        for i in range(self.time_div):
            elm_x = self._slice_x(idx+i)
            input_x[:,i] = elm_x
        target_y = self._slice_y(idx+self.time_div)

        return input_x, target_y

    def _slice_x(self, idx):
        row = self.pd[idx].split(',') # time, seocho, yongsan, namhyun, 256*256
        time, seocho, yongsan, namhyun = row[0:4]
        elm_x = np.array([seocho, yongsan, namhyun])

        return elm_x

    def _slice_y(self, idx):
        try:
            row = self.pd[idx].split(',') # time, seocho, yongsan, namhyun, 256*256
            val = row[4:]
            grid = np.zeros([self.dsize,self.dsize], dtype=np.float32)
            for i in range(self.dsize):
                tstr = val[i].strip().split(' ')
                if len(tstr) == 256:
                    grid[i] = np.array(list(map(float, tstr)))
            elm_y = grid.reshape(-1)
        except Exception as e:
            # print(traceback.format_exc())
            print(f'check (idx ,i): ({idx},{i})')
            raise e


        return elm_y

    def __len__(self):
        return self.len
