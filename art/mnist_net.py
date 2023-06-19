import torch
import torch.nn as nn
import torch.nn.functional as F
from diffabs import AbsDom, AbsEle
import numpy as np
import ast
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Iterable, Sequence, Union
from torch import Tensor
sys.path.append(str(Path(__file__).resolve().parent.parent))

class Mnist_net(nn.Module):
    '''
    abstract module of bank, credit and census
    # :param json file: The configuration file of Fairness task in Socrates
    :param means: The means of Dataset
    :param range: The range of Dataset
    # :param inputsize: The input size of NN, which is related to Dataset

    '''
    def __init__(self, dom: AbsDom) -> None:
        super().__init__()
        self.dom = dom
        self.conv1 = dom.Conv2d(1, 32, kernel_size=5)
        self.conv2 = dom.Conv2d(32, 64, kernel_size=5)
        self.maxpool = dom.MaxPool2d(2)
        self.relu = dom.ReLU()
        self.fc1 = dom.Linear(1024, 32)
        self.fc2 = dom.Linear(32, 10)
        # TODO: flatten
        self.flatten = dom.Flatten()
        self.sigmoid = dom.Sigmoid()

    def forward(self, x: Union[Tensor, AbsEle]) -> Union[Tensor, AbsEle]:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.relu(x)
        # x = torch.flatten(x, 1)
        # x = x.view(x.size[0], 1024)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
    def split(self):
        return nn.Sequential(
            self.conv1,
            self.maxpool,
            self.relu,
            self.conv2,
            self.maxpool,
            self.relu,
            # torch.flatten(x, 1),
            self.flatten,
            self.fc1
        ), nn.Sequential(
            
            self.fc2,
            self.sigmoid()
        )
