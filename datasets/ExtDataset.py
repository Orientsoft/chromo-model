# coding:utf-8
import os
import random
from PIL import Image

from .simple import SimpleDataset

class ExtDataset(SimpleDataset):
    def __init__(self, data_list, balance, transform=None):
        super(ExtDataset, self).__init__(data_list, balance, transform)
        self.name = 'ExtDataset'

    def __repr__(self):
        return self.name

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        return super().__getitem__(index)