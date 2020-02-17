# coding:utf-8
import torch
from torch.utils.data import *
import torchvision.transforms as transforms

from PIL import Image
import sys
import os

from .simple import SimpleDataset

class PUDataset(SimpleDataset):
    """Positive Unlabeled Learning Dataset.
       Change ground truth to 1 for positive and -1 for unlabeld.
    """
    def __init__(self,
        data_list,
        balance,
        transform=None):
        super(PUDataset, self).__init__(data_list, balance, transform)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        path, image, gt = item
        if gt == 0:
            gt = 1
        elif gt == 1:
            gt = -1
        item = (path, image, gt)
        return item