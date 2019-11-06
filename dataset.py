import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from tqdm import tqdm
import os

class TSDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.x = torch.FloatTensor([x[0].astype(np.float32) for x in data])
        self.y = torch.LongTensor([x[1]-1 for x in data])

    def __getitem__(self, i):
        x = self.x[i]
        y = self.y[i]
        return x, y
    
    def __len__(self):
        return len(self.x)