import sys
from pathlib import Path
from typing import List, Optional, Tuple, Iterable, Sequence, Union
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from torch import Tensor, nn
from diffabs import AbsDom, AbsEle
from argparse import Namespace
from acas import AcasNet
import logging
from art import exp, acas, utils
from repair_moudle import SupportNet,PatchNet,IntersectionNetSum
from exp_acas import AcasPoints,AcasArgParser
from timeit import default_timer as timer

RES_DIR = Path(__file__).resolve().parent.parent / 'results' / 'acas' / 'repair' / 'test_support'
RES_DIR.mkdir(parents=True, exist_ok=True)
REPAIR_MODEL_DIR = Path(__file__).resolve().parent.parent / 'model' / 'reassure_format'

def test_support(nid: acas.AcasNetID, args: Namespace):
    fpath = nid.fpath()
    net, bound_mins, bound_maxs = acas.AcasNet.load_nnet(fpath, args.dom, device)
    if args.reset_params:
        net.reset_parameters()
    logging.info(net)

    # TODO the number of repair neural network
    n_repair = 5
    # TODO the construction of support and patch network
    input_size = 5
    hidden_size = [10,10,10]
    support_lists = []
    patch_lists = []
    for i in range(n_repair):
        support = SupportNet(input_size=input_size, dom=args.dom, hidden_sizes=hidden_size,
            name = f'repair{i}')
        patch = PatchNet(input_size=input_size, dom=args.dom, hidden_sizes=hidden_size,
            name = f'repair{i}')
        support_lists.append(support)
        patch_lists.append(patch)
    
    repair_net = IntersectionNetSum(dom=args.dom, target_net=net, support_nets=support_lists, patch_nets=patch_lists, device = device)
    # TODO
    repair_net.load_state_dict(torch.load(str(REPAIR_MODEL_DIR / '2_9.pt')))
    testset = AcasPoints.load(nid, train=False, device=device)
    support_out = []
    for patch in patch_lists:
        outs = support(testset.inputs)
        support_out.append(outs)
    
    patch_out_tensor = torch.stack(support_out)
    

    
    return patch_out_tensor



def _run_test_support(nids: List[acas.AcasNetID], args: Namespace):
    logging.info('===== start test support network ======')
    for nid in nids:
        logging.info(f'For {nid}')
        outs = test_support(nid, args)

def test_patch_distribution(parser: AcasArgParser):
    defaults = {
        # 'start_abs_cnt': 5000,
        'batch_size': 100,  # to make it faster
        'min_epochs': 25,
        'max_epochs': 35
    }
    parser.set_defaults(**defaults)
    args = parser.parse_args()

    logging.info(utils.fmt_args(args))
    nids = acas.AcasNetID.goal_safety_ids(args.dom)
    if args.reassure_support_and_patch_combine:
        _run_test_support(nids, args)

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    test_defaults = {
        # 'exp_fn': 'test_goal_safety',
        'exp_fn': 'test_patch_distribution',
        'reassure_support_and_patch_combine': True
        # 'no_refine': True
    }
    parser = AcasArgParser(RES_DIR, description='ACAS Xu Correct by Construction')
    parser.set_defaults(**test_defaults)
    args, _ = parser.parse_known_args()

    exp_fn = locals()[args.exp_fn]
    start = timer()
    exp_fn(parser)

    logging.info(f'Total Cost Time: {timer() - start}s.\n\n\n')
    pass