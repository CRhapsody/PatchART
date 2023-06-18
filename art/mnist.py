'''
MNist task setup, includes :1. the class Photo property and its child(Mnist) 2. the class MNist net
'''
import datetime
import enum
import sys
from itertools import product
from pathlib import Path
from typing import List, Optional, Tuple, Iterable, Sequence, Union
from abc import ABC, abstractclassmethod

import torch
from torch import Tensor, nn
from diffabs import AbsDom, AbsEle
import numpy as np
import ast
sys.path.append(str(Path(__file__).resolve().parent.parent))

from art.prop import OneProp, AndProp
from art.utils import sample_points


class PhotoProp(OneProp):
    '''
    Define a fairness property
    incremental param:
    :param inputs : the data should be fairness
    :param protected_feature: the idx of input which should be protected
    '''
    def __init__(self, input_dimension: int, name: str, dom: Optional[AbsDom], safe_fn: str, viol_fn: str, fn_args: Iterable):
        super().__init__(name, dom, safe_fn, viol_fn, fn_args)
        self.input_dimension = input_dimension
        self.input_bounds = [(0, 1) for _ in range(input_dimension)]
    
    def lbub(self, device=None) -> Tuple[Tensor, Tensor]:
        """ Return <LB, UB>, both of size <1xDim0>. """
        bs = torch.tensor(self.input_bounds)
        bs = bs.unsqueeze(dim=0)
        lb, ub = bs[..., 0], bs[..., 1]
        if device is not None:
            lb, ub = lb.to(device), ub.to(device)
        return lb, ub
    
    def set_input_bound(self, idx: int, new_low: float = None, new_high: float = None):
        low, high = self.input_bounds[idx]
        if new_low is not None:
            low = max(low, new_low)

        if new_high is not None:
            high = min(high, new_high)

        assert(low <= high)
        self.input_bounds[idx] = (low, high)
        return

class MnistFeatureProp(PhotoProp):
    
    LABEL_NUMBER = 10

    class MnistOut(enum.IntEnum):
        '''
        the number from 0-9
        '''
        ZERO = 0
        ONE = 1
        TWO = 2
        THREE = 3
        FOUR = 4
        FIVE = 5
        SIX = 6
        SEVEN = 7
        EIGHT = 8
        NINE = 9

    def __init__(self, input_dimension: int, name: str, dom: Optional[AbsDom], safe_fn: str, viol_fn: str, fn_args: Iterable):
        '''
        :param input_dimension: the dimension of input/feature
        '''
        super().__init__(input_dimension, name, dom, safe_fn, viol_fn, fn_args)
    
    @classmethod
    def all_props(cls, dom: AbsDom, DataList: List[Tuple[Tensor, Tensor]], input_dimension: int, radius: float = 0.1, tasktype:str = 'attack_feature'):
        '''
        :param tasktype: the type of task, e.g. 'attack_feature' in mnist repair
        :param dom: the domain of input, e.g. Deeppoly
        :param DataList: the list of data, e.g. [(data1, label1), (data2, label2), ...]
        :param input_dimension: the dimension of input/feature
        :param radius: the radius of the attack input/feature
        '''

        datalen = len(DataList)
        names = [(DataList[i][0], DataList[i][1]) for i in range(datalen)]
        a_list = []
        for data,label in names:
            a = getattr(cls, tasktype)(dom, input_dimension, data, label, radius)
            a_list.append(a)
        
        return a_list
    
    @classmethod
    def attack_feature(cls, tasktype: str, dom: AbsDom, input_dimension: int, data: Tensor, label: int, radius: float):
        '''
        The mnist feature property is Data-based property. One data point correspond to one l_0 ball.
        :params input_dimension: the input/feature dimension
        :params label: the output which should be retained
        :params radius: the radius of the attack input/feature
        '''
        p = MnistFeatureProp(name=tasktype, input_dimension=input_dimension, dom=dom, safe_fn='cols_is_max', viol_fn='col_not_max',
                    fn_args=[label])  # mean/range hardcoded 
        for j in range(input_dimension):
            p.set_input_bound(j, new_low=data[j].item() - radius)
            p.set_input_bound(j, new_high=data[j].item() + radius)     
                

        return p


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
        self.sigmoid = dom.sigmoid(dim=1)

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


    
    
    





