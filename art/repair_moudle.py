import sys
from pathlib import Path
from typing import List, Optional, Tuple, Iterable, Sequence, Union
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from torch import Tensor, nn
from diffabs import AbsDom, AbsEle

from acas import AcasNet

class SupportNet(nn.Module):
    '''
    Construct the support network for repair.
    (provisional:)
    The construction of it is full connection network. Its input is the input of neural networks.
    '''
    def __init__(self, input_size: int, dom :AbsDom, hidden_sizes: List[int],
                name: str, output_size: int ) -> None:
        '''
        :param hidden_sizes: the size of all hidden layers
        :param output_size: Due to the support network is characteristic function, the output of support network should be confidence.
        :param name: the name of this support network; maybe the repairing property belonging to it in the later
        '''
        super().__init__()
        self.name = name
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.n_layers = len(hidden_sizes) + 1

        # layer
        self.acti = dom.ReLU()
        self.all_linears = nn.ModuleList()
        in_sizes = [self.input_size] + self.hidden_sizes
        out_sizes = self.hidden_sizes + [self.output_size]
        for in_size, out_size in zip(in_sizes, out_sizes):
            self.all_linears.append(dom.Linear(in_size, out_size))
        return 
    
    def forward(self, x):
        for lin in self.all_linears[:-1]:
            x = lin(x)
            x = self.acti(x)
            
        # TODO last layer can be relu layer
        x = self.all_linears[-1](x)
        return x
        
    
    def __str__(self):
        """ Just print everything for information. """
        ss = [
            '--- SupportNet ---',
            'Name: %s' % self.name,
            'Num layers: %d (i.e. hidden + output, excluding input layer)' % self.n_layers,
            'Input size: %d' % self.input_size,
            'Hidden sizes (len %d): ' % len(self.hidden_sizes) + str(self.hidden_sizes),
            'Output size: %d' % self.output_size,
            'Activation: %s' % self.acti,
            '--- End of SupportNet ---'
        ]
        return '\n'.join(ss)



class PatchNet(nn.Module):
    '''
    Construct the patch network for repair.
    1. The Patchnet and Supportnet has one-to-one correspondence
    (provisional:)
    The construction of it is full connection network. Its input is the input of neural networks.
    '''
    def __init__(self, input_size: int, dom :AbsDom, hidden_sizes: List[int],
                name: str, output_size=5 ) -> None:
        '''
        :param hidden_sizes: the size of all hidden layers
        :param output_size: The patch network directly add to the output , and its input is the input of neural network. So its outputsize should be equal to the orignal outputsize
        :param name: the serial number of this support network; maybe the repairing property belonging to it in the later
        '''
        super().__init__()
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.n_layers = len(hidden_sizes) + 1

        # layer
        self.acti = dom.ReLU()
        self.all_linears = nn.ModuleList()
        in_sizes = [self.input_size] + self.hidden_sizes
        out_sizes = self.hidden_sizes + [self.output_size]
        for in_size, out_size in zip(in_sizes, out_sizes):
            self.all_linears.append(dom.Linear(in_size, out_size))
        return
    
    def forward(self, x):
        for lin in self.all_linears[:-1]:
            x = lin(x)
            x = self.acti(x)
            
        x = self.all_linears[-1](x)
        return x

    
    def __str__(self):
        """ Just print everything for information. """
        ss = [
            '--- PatchNet ---',
            'Name: %s' % self.name,
            'Num layers: %d (i.e. hidden + output, excluding input layer)' % self.n_layers,
            'Input size: %d' % self.input_size,
            'Hidden sizes (len %d): ' % len(self.hidden_sizes) + str(self.hidden_sizes),
            'Output size: %d' % self.output_size,
            'Activation: %s' % self.acti,
            '--- End of PatchNet ---'
        ]
        return '\n'.join(ss)

class Netsum(nn.Module):
    '''
    This class is to add the patch net to target net:
    
    '''
    def __init__(self, dom: AbsDom, target_net: AcasNet, patch_nets: List[nn.Module], device = None, ):
        '''
        :params 
        '''
        super().__init__()
        self.target_net = target_net
        
        # for support, patch in zip(support_nets, patch_nets):
        #     assert(support.name == patch.name), 'support and patch net is one-to-one'

        # self.support_nets = support_nets
        self.patch_nets = patch_nets
        self.acti = dom.ReLU()
        self.len_patch_lists = len(self.patch_nets)

        if device is not None:
            for i,patch in enumerate(self.patch_nets):
                self.add_module(f'patch{i}',patch)
                patch.to(device)
        
        
        # self.connect_layers = []



    def forward(self, x):
        out = self.target_net(x)
        for i,patch in enumerate(self.patch_nets):
            out += self.acti(patch(x) + self.K[i]*support(x) - self.K[i]) \
                - self.acti(-1*patch(x) + self.K[i]*support(x) - self.K[i])
        return out
    
    def __str__(self):
        """ Just print everything for information. """
        # TODO information for each support and patch net as components
        ss = [
            '--- IntersectionNetSum ---',
            'Num net: support %d , patch %d' % (len(self.support_nets),len(self.patch_nets)),
            'Input size: %d' % self.target_net.input_size,
            'Output size: %d' % self.target_net.output_size,
            'Threshold value: %d' % self.k,
            '--- End of IntersectionNetSum ---'
        ]
        return '\n'.join(ss)


class IntersectionNetSum(nn.Module):
    '''
    This class is the complement of single-region repair in REASSURE.The function is:
    
     h_{\mathcal{A}}(x, \gamma)=\sigma\left(p_{\mathcal{A}}(x)+K \cdot g_{\mathcal{A}}(x, \gamma)-K\right)-\sigma\left(-p_{\mathcal{A}}(x)+K \cdot g_{\mathcal{A}}(x, \gamma)-K\right)
    '''
    def __init__(self, dom: AbsDom, target_net: AcasNet, support_nets : List[nn.Module], patch_nets: List[nn.Module], device = None, ):
        '''
        :params K : we define the threshold value k as 1e8 as default.
        '''
        super().__init__()
        self.target_net = target_net
        
        for support, patch in zip(support_nets, patch_nets):
            assert(support.name == patch.name), 'support and patch net is one-to-one'

        self.support_nets = support_nets
        self.patch_nets = patch_nets
        self.acti = dom.ReLU()
        self.len_support_lists = len(self.support_nets)
        self.K = [0.01 for i in range(self.len_support_lists)] # initial

        if device is not None:
            for i,support, patch in zip(range(len(self.support_nets)),self.support_nets, self.patch_nets):
                self.add_module(f'support{i}',support)
                support.to(device)
                self.add_module(f'patch{i}',patch)
                patch.to(device)
        
        
        # self.connect_layers = []



    def forward(self, x):
        out = self.target_net(x)
        for i, support, patch in zip(range(self.len_support_lists),self.support_nets, self.patch_nets):
            out += self.acti(patch(x) + self.K[i]*support(x) - self.K[i]) \
                - self.acti(-1*patch(x) + self.K[i]*support(x) - self.K[i])
        return out
    
    def __str__(self):
        """ Just print everything for information. """
        # TODO information for each support and patch net as components
        ss = [
            '--- IntersectionNetSum ---',
            'Num net: support %d , patch %d' % (len(self.support_nets),len(self.patch_nets)),
            'Input size: %d' % self.target_net.input_size,
            'Output size: %d' % self.target_net.output_size,
            'Threshold value: %d' % self.k,
            '--- End of IntersectionNetSum ---'
        ]
        return '\n'.join(ss)


class ConnectionNetSum(nn.Module):
    '''
    This class is to directly connect the support net and patch net:
    
    '''
    def __init__(self, dom: AbsDom, target_net: AcasNet, support_nets : List[nn.Module], patch_nets: List[nn.Module], device = None, ):
        '''
        :params
        '''
        super().__init__()
        self.target_net = target_net
        
        for support, patch in zip(support_nets, patch_nets):
            assert(support.name == patch.name), 'support and patch net is one-to-one'

        self.support_nets = support_nets
        self.patch_nets = patch_nets
        self.acti = dom.ReLU()
        self.len_support_lists = len(self.support_nets)

        if device is not None:
            for i,support, patch in zip(range(len(self.support_nets)),self.support_nets, self.patch_nets):
                self.add_module(f'support{i}',support)
                support.to(device)
                self.add_module(f'patch{i}',patch)
                patch.to(device)
        
        
        # self.connect_layers = []



    def forward(self, x):
        out = self.target_net(x)
        for i, support, patch in zip(range(self.len_support_lists),self.support_nets, self.patch_nets):
            out += self.acti(patch(x) + self.K[i]*support(x) - self.K[i]) \
                - self.acti(-1*patch(x) + self.K[i]*support(x) - self.K[i])
        return out
    
    def __str__(self):
        """ Just print everything for information. """
        # TODO information for each support and patch net as components
        ss = [
            '--- IntersectionNetSum ---',
            'Num net: support %d , patch %d' % (len(self.support_nets),len(self.patch_nets)),
            'Input size: %d' % self.target_net.input_size,
            'Output size: %d' % self.target_net.output_size,
            'Threshold value: %d' % self.k,
            '--- End of IntersectionNetSum ---'
        ]
        return '\n'.join(ss)