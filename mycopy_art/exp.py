""" Common utilities and functions used in multiple experiments. """

import argparse
import logging
import time
from pathlib import Path
from diffabs import DeeppolyDom, IntervalDom
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import numpy as np
import random

class ExpArgParser(argparse.ArgumentParser):
    """ Override constructor to add more arguments or modify existing defaults.
        :param log_dir: if not None, the directory for log file to dump, the log file name is fixed
    """
    def __init__(self, log_dir, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self.log_dir = log_dir

        # expriment hyper-parameters
        self.add_argument('--exp_fn', type=str, 
                            help = 'the experiment function to run')
        self.add_argument('--seed', type=int, default=None,
                            help='the random seed for all')
        #art hyper-parameters
        self.add_argument('--dom', type=str, choices=['deeppoly', 'interval'], default='deeppoly',
                            help='the abstract domain to use')  
        self.add_argument('--start_abs_cnt', type=int, default=5000,
                            help='do some refinement before training to have more training data')
        self.add_argument('--max_abs_cnt', type=int, default=10000,
                            help='stop refinement after exceeding this many abstractions')
        # TODO 这个参数没看懂，等着看看论文
        self.add_argument('--refine_top_k', type=int, default=200,
                            help='select top k abstractions to refine every time') 
        self.add_argument('--tiny_width', type= float,default=1e-3,
                            help='refine a dimension only when its width still > this tiny_width')

        # training hyper-parameters
        self.add_argument('--lr',type=float,default=1e-3,
                            help='initial learning rate during training')
        self.add_argument('--batch_size', type=int, default=32,
                          help='mini batch size during each training epoch')
        self.add_argument('--min_epochs', type=int, default=90,
                          help='at least run this many epochs for sufficient training')
        self.add_argument('--max_epochs', type=int, default=100,
                          help='at most run this many epochs before too long')

        # training flags
        self.add_argument('--use_scheduler', action='store_true', default=False,
                            help='using learning rate scheduler during training')
        self.add_argument('--no_pts', action='store_true', default=False,
                          help='not using concrete sampled/prepared points during training')
        self.add_argument('--no_abs', action='store_true', default=False,
                          help='not using abstractions during training')
        self.add_argument('--no_refine', action='store_true', default=False,
                          help='disable refinement during training')
                                                                
        # printing flags
        group = self.add_argument_group()
        group.add_argument('--quiet', action='store_true', default=False,
                        help='show warning level logs (default: info)')
        group.add_argument("--debug", action="store_true", default=False,
                           help='show debug level logs (default: info)')
        return

    def parse_args(self,args = None, namespace = None): # return a namespace
        res = super().parse_args(args, namespace)
        self.setup_logger(res)
        if res.seed is not None:
            random_seed(res.seed)
        self.setup_rest(res)
        return res
        
    
    def setup_logger(self, args: argparse.Namespace): 
        logger = logging.getLogger()
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s]%(message)s')

        if args.quiet == True:
            # default to be warning level
            pass
        elif args.debug == True: 
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
    
        #以时间戳+exp_fn为名将log信息实时传送到log文件，需要用filehandle处理
        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        args.stamp = f'{args.exp_fn}-{timestamp}'
        logger.handlers = []  # reset, otherwise it may duplicate many times when calling setup_logger() multiple times
        if self.log_dir is not None:
            log_Path = Path(self.log_dir, f'{args.stamp}.log')
            file_handler = logging.FileHandler(filename=log_Path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        #还需要单独创建一个流handler, 用来将log打在终端上
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return



    
    
    def setup_rest(self, args: argparse.Namespace): 
        '''
        Override this method to set up those not easily specified via command line arguments.
        '''
        
        assert not (args.no_pts and args.no_abs), 'training what?' #不清楚这个注释为什么可以这么接，等着看一下
        args.dom = {
            'deeppoly': DeeppolyDom(),
            'interval': IntervalDom()
        }[args.dom]

        if args.use_scheduler:
            args.scheduler_fn = lambda opti : ReduceLROnPlateau(optimizer=opti)
        else:
            args.scheduler_fn = lambda opti : None
        return
    
    
def random_seed(seed):
    '''
    set random seed for all
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return

                            
