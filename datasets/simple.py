import torch
from torch.utils.data import *
import torchvision.transforms as transforms

from PIL import Image
import sys
import os
import random

class SimpleDataset(Dataset):
    def __init__(self, data_list, balance=None, transform=None):
        """
        Args:
            data_list: str, path, data list format 'path, gt\n'
            balance: up or down or no
        
        self.data = [(path, class_idx), ...]
        """
        self.data_list = data_list
        self.data = []
        self.data_len = 0
        self.balance = balance
        self.transform = transform

        with open(data_list) as f:
            lines = f.readlines()
        if 'gt' in lines[0]: lines = lines[1:]
        for line in lines:
            path, idx = line.strip('\n').split(', ')
            self.data.append((path, int(idx)))

        if self.balance is not None:
            self._balance()

    def __getitem__(self, idx):
        path, gt = self.data[idx]
        image = Image.open(path)
        image = self.transform(image) 
        return path, image, gt

    def __len__(self):
        return len(self.data)

    def _balance(self):
        """balance data.
        Returns:
            cls2num: {int(idx):int(num)}
        """
        cls2path = {}
        cls2num = {}
        for path, idx in self.data:
            if idx not in cls2path:
                cls2path[idx] = [path]
            else:
                cls2path[idx].append(path)
        for idx, path_list in cls2path.items():
            cls2num[idx] = len(path_list)
        if self.balance == 'up':
            target_cls_num = max(cls2num.values())
        elif self.balance == 'down':
            target_cls_num = min(cls2num.values())
        for idx in cls2path:
            path_list = cls2path[idx][:]
            if self.balance == 'up':
                while len(path_list) < target_cls_num:
                    path_list.append(random.choice(cls2path[idx]))
            elif self.balance == 'down':
                path_list = path_list[:target_cls_num]
            cls2path[idx] = path_list
        self.data = []
        for idx, path_list in cls2path.items():
            for path in path_list:
                self.data.append((path, idx))

class _SimpleDataset(Dataset):
    # NOT USED
    def __init__(
        self,
        root,
        phase='train',
        transform=None):
        self.root = root
        self.phase = phase
        self.transform = transform

        self.num_classes = 0
        self.samples = []
        self.max_class_count = 0
        self.total_len = 0

        if self.phase == 'train': # balance data in train phase
            classes = os.listdir(os.path.join(self.root, self.phase))
            classes.sort()
            self.num_classes = len(classes)

            for item in classes:
                class_path = os.path.join(self.root, self.phase, item)

                pics = os.listdir(class_path)
                pic_paths = [os.path.join(class_path, pic) for pic in pics]
                pic_paths.sort()
                self.samples.append([(pic_path, int(item)) for pic_path in pic_paths])
            
            # for data balance between classes
            class_counts = [len(class_sample) for class_sample in self.samples]
            class_counts.sort()
            self.max_class_count = class_counts[-1]
            
            self.total_len = self.max_class_count * self.num_classes
        elif self.phase == 'up balance val':
            classes = os.listdir(os.path.join(self.root, 'val'))
            classes.sort()
            self.num_classes = len(classes)

            for item in classes:
                class_path = os.path.join(self.root, 'val', item)

                pics = os.listdir(class_path)
                pic_paths = [os.path.join(class_path, pic) for pic in pics]
                pic_paths.sort()
                self.samples.append([(pic_path, int(item)) for pic_path in pic_paths])
            
            # for data balance between classes
            class_counts = [len(class_sample) for class_sample in self.samples]
            class_counts.sort()
            self.max_class_count = class_counts[-1]
            
            self.total_len = self.max_class_count * self.num_classes
        elif self.phase == 'down balance val': # balance the val
            classes = os.listdir(os.path.join(self.root, 'val'))
            classes.sort()

            for item in classes:
                class_path = os.path.join(self.root, 'val', item)

                pics = os.listdir(class_path)
                pic_paths = [os.path.join(class_path, pic) for pic in pics]
                if item == '1':
                    pic_paths = pic_paths[:500]
                pic_paths.sort()

                self.samples.append([(pic_path, int(item)) for pic_path in pic_paths])
                self.total_len += len(pic_paths)
        elif self.phase == 'fake':
            classes = os.listdir(os.path.join(self.root, 'train'))
            classes.sort()

            for item in classes:
                class_path = os.path.join(self.root, 'train', item)

                pics = os.listdir(class_path)
                pic_paths = [os.path.join(class_path, pic) for pic in pics[:500]]
                pic_paths.sort()

                self.samples.append([(pic_path, int(item)) for pic_path in pic_paths])
                self.total_len += len(pic_paths)
        else: # test or val
            classes = os.listdir(os.path.join(self.root, self.phase))
            classes.sort()

            for item in classes:
                class_path = os.path.join(self.root, self.phase, item)

                pics = os.listdir(class_path)
                pic_paths = [os.path.join(class_path, pic) for pic in pics]
                pic_paths.sort()

                self.samples.append([(pic_path, int(item)) for pic_path in pic_paths])
                self.total_len += len(pic_paths)

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        if self.phase == 'train':
            class_index = index // self.max_class_count
            item_index = (index % self.max_class_count) % len(self.samples[class_index])

            image_path, gt = self.samples[class_index][item_index]

            image = Image.open(image_path)
            
            if self.transform is not None:
                image = self.transform(image)

            return image_path, image, gt
        elif self.phase == 'up balance val':
            class_index = index // self.max_class_count
            item_index = (index % self.max_class_count) % len(self.samples[class_index])

            image_path, gt = self.samples[class_index][item_index]

            image = Image.open(image_path)
            
            if self.transform is not None:
                image = self.transform(image)

            return image_path, image, gt
        elif self.phase == 'fake':
            total_count = 0

            for class_index, item in enumerate(self.samples):
                if total_count <= index < total_count + len(item):
                    item_index = index - total_count

                    sample = item[item_index]
                    image_path, gt = sample

                    image = Image.open(image_path)

                    if self.transform is not None:
                        image = self.transform(image)

                    return image_path, image, gt
                else:
                    total_count += len(item)
        else: # test or val
            total_count = 0

            for class_index, item in enumerate(self.samples):
                if total_count <= index < total_count + len(item):
                    item_index = index - total_count

                    sample = item[item_index]
                    image_path, gt = sample

                    image = Image.open(image_path)

                    if self.transform is not None:
                        image = self.transform(image)

                    return image_path, image, gt
                else:
                    total_count += len(item)
