#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class Net12(nn.Module):
    def __init__(self):
        super(Net12,self).__init__()
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=(3,3), stride=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=(3,3), stride=2)
        self.fc = torch.nn.Linear(4 * 4 * 16, 2)
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class Net24(nn.Module):
    def __init__(self):    
        super(Net24, self).__init__()
    def forward(self, x):
        return x
    