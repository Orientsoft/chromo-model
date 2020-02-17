# coding:utf-8
import torch
import torch.nn as nn

from .resnet import *

class Identity(nn.Module):
    def __init__(self):
        pass
    def forward(self, x):
        return x


class Bilinear(nn.Module):
    def __init__(self, pretrained, num_classes=2, base='resnet18'):
        """from github.com/HaoMood/blinear-cnn-faster
        Args:
            base: 基础模型, output is (bs, 512, 1, 1)
        """
        super(Bilinear, self).__init__()
        self.feature = globals()[base](pretrained, num_classes=num_classes)
        #self.feature = getattr(resnet, base)(pretrained, num_classes=num_classes)
        self.feature = torch.nn.Sequential(*list(self.feature.children())[:-2]) # 去掉avgpool和fc

        self.fc = torch.nn.Linear(in_features=512 * 512, out_features=num_classes, bias=True)
    
    def forward(self, x):
        """
        Args:
            x: tensor, (bs, 1, 448, 448)
        """
        N = x.size()[0]
        x = self.feature(x) # (bs, 512, 14, 14)
        x = torch.reshape(x, (N, 512, 14 * 14))
        x = torch.bmm(x, torch.transpose(x, 1, 2)) / (14 * 14) # (bs, 512, 512)
        x = torch.reshape(x, (N, 512 * 512))

        # normalization
        x = torch.sqrt(x + 1e-5)
        x = torch.nn.functional.normalize(x)

        # Classification.
        x = self.fc(x)
        return x

def bi18(pretrained, num_classes=2):
    return Bilinear(pretrained, num_classes=num_classes, base='resnet18')

def bi34(pretrained, num_classes=2):
    return Bilinear(pretrained, num_classes=num_classes, base='resnet34')

def bi50(pretrained, num_classes=2):
    return Bilinear(pretrained, num_classes=num_classes, base='resnet50')

def bi101(pretrained, num_classes=2):
    return Bilinear(pretrained, num_classes=num_classes, base='resnet101')