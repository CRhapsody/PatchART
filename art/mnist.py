'''
Fairness task setup, includes :1. the class Fairness property and its child(bank, credit, census) 2. the class Fairnessnet and its child
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
# from exp_fairness import FairnessPoint


CAUSAL_DIR = Path(__file__).resolve().parent.parent / 'data' / 'causal'
CAUSAL_DATA_DIR = CAUSAL_DIR / 'ptdata'
CAUSAL_MODEL_DIR = CAUSAL_DIR / 'origin_model'
BANK_DIR = CAUSAL_DIR / 'bank'
CREDIT_DIR = CAUSAL_DIR / 'credit'
CENSUS_DIR = CAUSAL_DIR / 'census'
# BANK_DIR_DATA = CAUSAL_DIR / 'bank' / 'data'
# CREDIT_DIR_DATA = CAUSAL_DIR / 'credit' / 'data'
# CENSUS_DIR_DATA = CAUSAL_DIR / 'census' / 'data'
# BANK_DIR_MODEL = CAUSAL_DIR / 'bank' / 'nnet' / 'model.txt'
# CREDIT_DIR_MODEL = CAUSAL_DIR / 'credit' / 'nnet' / 'model.txt'
# CENSUS_DIR_MODEL = CAUSAL_DIR / 'census' / 'nnet' / 'model.txt'

# bank_means = [3.645860433578491, 6.01815938949585, 0.6809626221656799, 1.6988564729690552, 0.01802658662199974, 23.792396545410156, 0.5558381676673889, 0.1602264940738678, 1.3597575426101685, 15.806418418884277, 5.144655227661133, 24.545928955078125, 2.763840675354004, 0.817367434501648, 0.18263255059719086, 0.35776692628860474]
# credit_means = [1.0383332967758179, 20.219999313354492, 2.4000000953674316, 2.638333320617676, 31.481666564941406, 0.8149999976158142, 2.2850000858306885, 2.986666679382324, 0.3333333432674408, 0.15833333134651184, 2.878333330154419, 1.3516666889190674, 3.0833332538604736, 1.653333306312561, 0.9100000262260437, 1.3849999904632568, 1.8883333206176758, 1.159999966621399, 0.3683333396911621, 0.0533333346247673]
# census_means = [3.410552501678467, 6.334203720092773, 8.986978530883789, 3.4030895233154297, 1.0774853229522705, 10.127361297607422, 2.380854368209839, 0.45975247025489807, 0.6692054867744446, 1.0428426265716553, 0.8501274585723877, 40.43745422363281, 3.270876169204712]

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
        self.input_bounds = [(0, 10) for i in range(input_dimension)]
    
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
        # self.input_bounds = [(0,3),(1,80),(0,4),
        #                       (0,10),(1,200),(0,4),
        #                       (0,4),(1,4),(0,1),
        #                       (0,2),(1,4),(0,3),
        #                       (1,8),(0,2),(0,2),
        #                       (1,4),(0,3),(1,2),
        #                       (0,1),(0,1)]
        
    
    @classmethod
    def all_props(cls, tasktype:str, dom: AbsDom, DataList: Tuple[Tensor, Tensor],input_dimension: int):
        '''
        In Socrates, Credit task has two Prop: age and gender
        We guess elements in Datalist is tuple with (data, protected)
        :params taskype: Dataset has tasktype, which is the protected property
        :params Datalist: (inputs, label) : (Tensor, Tensor), but overwrite in class PseudoLenDataset
        '''

        datalen = len(DataList)
        names = [(DataList[i][0], DataList[i][1]) for i in range(datalen)]
        a_list = []
        for data,label in names:
            a = getattr(cls, tasktype)(dom, input_dimension, data, label)
            a_list.extend(a)
        
        return a_list
        # a = [getattr(cls, tasktype)(dom, data, label) for data,label in names]
        # return a
    
    @classmethod
    # def age(cls, dom: AbsDom, data: Tensor, label: int):
    #     '''
    #     The fairness property is Data-based property. One data point correspond to one property.
    #     :params label: the output which should be retained
    #     '''
    #     p = CreditProp(name='age', input_dimension=CreditProp.INPUT_DIMENSION, dom=dom, safe_fn='cols_is_max', viol_fn='col_not_max',
    #                 fn_args=[label])  # mean/range hardcoded 
    #     for j in range(CreditProp.INPUT_DIMENSION):      
    #         if j == CreditProp.Protected.AGE:
    #             p.set_input_bound(CreditProp.Protected.AGE, new_low=1)
    #             p.set_input_bound(CreditProp.Protected.AGE, new_high=8)
    #         else:                
    #             p.set_input_bound(j, new_low=data[j].item())
    #             p.set_input_bound(j, new_high=data[j].item())

    #     return p


    def attack_feature(cls, dom: AbsDom, input_dimension: int, data: Tensor, label: int):
        '''
        The fairness property is Data-based property. One data point correspond to one property.
        :params input_dimension: the input/feature dimension
        :params label: the output which should be retained
        '''
        p_list = []
        # (1, 8)
        for i in range(1, 9):
            p = MnistFeatureProp(name='attack_feature', input_dimension=input_dimension, dom=dom, safe_fn='cols_is_max', viol_fn='col_not_max',
                     fn_args=[label])  # mean/range hardcoded 
            for j in range(input_dimension):      
                if j == CreditProp.Protected.AGE:
                    p.set_input_bound(CreditProp.Protected.AGE, new_low=i)
                    p.set_input_bound(CreditProp.Protected.AGE, new_high=i)
                else:                
                    p.set_input_bound(j, new_low=data[j].item())
                    p.set_input_bound(j, new_high=data[j].item())
            p_list.append(p)

        return p_list

    @classmethod
    # def gender(cls, dom: AbsDom, data: Tensor, label: int):
    #     '''
    #     The fairness property is Data-based property. One data point correspond to one property.
    #     :params protected: the output which should be retained
    #     '''
    #     p = CreditProp(name='gender', input_dimension=CreditProp.INPUT_DIMENSION, dom=dom, safe_fn='cols_is_max', viol_fn='col_not_max',
    #                  fn_args=[label])  # mean/range hardcoded
    #     for i in range(CreditProp.INPUT_DIMENSION):
    #         if i == CreditProp.Protected.GENDER:
    #             p.set_input_bound(CreditProp.Protected.GENDER, new_low=0)
    #             p.set_input_bound(CreditProp.Protected.GENDER, new_high=1)
    #         else:                
    #             p.set_input_bound(i, new_low=data[i].item())
    #             p.set_input_bound(i, new_high=data[i].item())

    #     return p

class FairnessNet(nn.Module):
    '''
    abstract module of bank, credit and census
    # :param json file: The configuration file of Fairness task in Socrates
    :param means: The means of Dataset
    :param range: The range of Dataset
    # :param inputsize: The input size of NN, which is related to Dataset

    '''
    def __init__(self, dom: AbsDom, input_size, output_size, hidden_sizes: List[int], means: List[float] = None, ranges: List[float] = None) -> None:
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.n_layers = len(hidden_sizes) + 1

        #initial, but means defined in their chlid classes respectively
        self.means = means if means is not None else [0.0] * (self.input_size + 1)
        self.ranges = ranges if ranges is not None else [1.0] * (self.input_size + 1)

        self.acti = dom.ReLU()
        self.all_linears = nn.ModuleList()
        in_sizes = [self.input_size] + self.hidden_sizes
        out_sizes = self.hidden_sizes + [self.output_size]
        for in_size, out_size in zip(in_sizes, out_sizes):
            self.all_linears.append(dom.Linear(in_size, out_size))
        return


    
    def normalize_inputs(self, t: Tensor, mins: Sequence[float], maxs: Sequence[float]) -> Tensor:
        """ Normalize: ([min, max] - mean) / range """
        slices = []
        for i in range(self.input_size):
            slice = t[:, i:i+1]
            slice = slice.clamp(mins[i], maxs[i])
            # slice -= self.means[i]
            # slice /= self.ranges[i]
            slices.append(slice)
        return torch.cat(slices, dim=-1)

    def forward(self, x: Union[Tensor, AbsEle]) -> Union[Tensor, AbsEle]:
        """ Normalization and Denomalization are called outside this method. """
        for lin in self.all_linears[:-1]:
            x = lin(x)
            x = self.acti(x)

        x = self.all_linears[-1](x)
        return x
    
    @classmethod
    def load_parameters(cls, filepath: str, dom: AbsDom):

        
        _num_layers = 0  # Number of layers in the network (excluding inputs, hidden + output = num_layers).
        _input_size = 0  # Number of inputs to the network.
        _output_size = 0
        _layer_sizes = []
        _layer_weights = []
        _layer_biases = []
        input = open(filepath, 'r')
        lines = input.readlines()


        for i in range(1, (int((len(lines)-1)/2))+1):
            _num_layers += 1

            wline = 2*i -1
            bline = 2*i
            w = np.array(ast.literal_eval(lines[wline]))
            b = np.array(ast.literal_eval(lines[bline]))

            if i == 1: #first layer
                _input_size = w.shape[0]
                _layer_sizes.append(_input_size)
            elif i == ((len(lines)-1)/2): # output_layer
                _output_size = w.shape[1]
                _layer_sizes.append(w.shape[0])
                _layer_sizes.append(w.shape[1])

            else:
                _layer_sizes.append(w.shape[0])

            w = w.transpose(1,0) # The linear type in torch is (out_feature, input_feature)
            _layer_weights.append(torch.tensor(w,dtype=torch.float32))
            _layer_biases.append(torch.tensor(b,dtype=torch.float32))

        assert _layer_sizes[0] == _input_size
        assert _layer_sizes[-1] == _output_size
        _hidden_sizes = _layer_sizes[1:-1]

        return _input_size, _output_size, _hidden_sizes, _layer_weights, _layer_biases
    

class BankNet(FairnessNet):
    '''
    The fairness task —— bank,
    '''
    def __init__(self, dom: AbsDom, input_size, output_size, hidden_sizes: List[int], means: List[float] = None, ranges: List[float] = None) -> None:
        super().__init__(dom, input_size, output_size, hidden_sizes, means, ranges)

class CreditNet(FairnessNet):
    '''
    The fairness task —— Credit
    '''
    def __init__(self, dom: AbsDom, input_size, output_size, hidden_sizes: List[int], means: List[float] = None, ranges: List[float] = None) -> None:
        super().__init__(dom, input_size, output_size, hidden_sizes)
        self.inputs_bounds = [(0,3),(1,80),(0,4),
                              (0,10),(1,200),(0,4),
                              (0,4),(1,4),(0,1),
                              (0,2),(1,4),(0,3),
                              (1,8),(0,2),(0,2),
                              (1,4),(0,3),(1,2),
                              (0,1),(0,1)]
        self.means = [1.0383332967758179, 20.219999313354492, 2.4000000953674316, 2.638333320617676, 31.481666564941406, 0.8149999976158142, 2.2850000858306885, 2.986666679382324, 0.3333333432674408, 0.15833333134651184, 2.878333330154419, 1.3516666889190674, 3.0833332538604736, 1.653333306312561, 0.9100000262260437, 1.3849999904632568, 1.8883333206176758, 1.159999966621399, 0.3683333396911621, 0.0533333346247673]
        assert(len(self.means) == 20)
        assert(input_size == 20)
        self.bound_mins = [mins for mins,_ in self.inputs_bounds]
        self.bound_maxs = [maxs for _,maxs in self.inputs_bounds]
        self.ranges = [maxs - mins for mins, maxs in self.inputs_bounds]
    
    @classmethod
    def load_net(cls,filepath: str, dom: AbsDom, device)-> Tuple[FairnessNet, List[float], List[float]]:
        _input_size, _output_size, _hidden_sizes, _layer_weights, _layer_biases = \
            FairnessNet.load_parameters(filepath, dom)
        # ===== Use the parsed information to build AcasNet =====
        # TODO substitute the means and ranges with correspond means and range
        net = CreditNet(dom, _input_size, _output_size, _hidden_sizes)

        
        # === populate weights and biases ===
        assert len(net.all_linears) == len(_layer_weights) == len(_layer_biases)
        for i, linear in enumerate(net.all_linears):
            linear.weight.data = _layer_weights[i]
            linear.bias.data = _layer_biases[i]

        if device is not None:
            net = net.to(device)
        return net, net.bound_mins, net.bound_maxs

    def __str__(self):
        """ Just print everything for information. """
        ss = [
            '---CreditNet ---',
            'Num layers: %d (i.e. hidden + output, excluding input layer)' % self.n_layers,
            'Input size: %d' % self.input_size,
            'Hidden sizes (len %d): ' % len(self.hidden_sizes) + str(self.hidden_sizes),
            'Output size: %d' % self.output_size,
            'Means for scaling (len %d): ' % len(self.means) + str(self.means),
            'Ranges for scaling (len %d): ' % len(self.ranges) + str(self.ranges),
            'Activation: %s' % self.acti,
            '--- End of CreditNet ---'
        ]
        return '\n'.join(ss)

    
    
    





