import torch
import exp
from typing import Optional
from timeit import default_timer as timer
import logging
import utils
import acas
from pathlib import Path

RES_DIR = Path(__file__).resolve().parent.parent / 'results' / 'acas_res'
RES_DIR.mkdir(parents= True, exist_ok= True)


class AcasArgParser(exp.ExpArgParser):
    def __init__(self, log_dir: Optional[str], *arg, **kwargs):
        super().__init__(log_dir, *arg, **kwargs)


def test_goal_safety(parser: AcasArgParser):
    default = {
        'batch_size':100,
        'min_epochs':25,
        'max_epochs':35
    }
    parser.set_defaults(**default)
    args = parser.parse_args()

    logging.info(utils.fmt_args(args))
    nids = acas.AcasNetID.goal_safety_ids(args.dom)
    _run(nids, args)

    pass

def _run(nids : acas.AcasNetID,args):
    # TODO
    pass

if __name__ =="__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    test_exp = {
        'exp_fn':'test_goal_safety'
    }
    parser = AcasArgParser(RES_DIR, description='ACAS Xu Correct by Construction')
    parser.set_defaults(**test_exp)

    # 这里可能不能用parse_args, 因为还没有真正调用训练
    # args = parser.parse_args()

    args, _ = parser.parse_known_args()
    exp_fn = locals()[args.exp_fn]
    start = timer()
    exp_fn(parser)

    logging.info(f'Total Cost Time: {timer() - start}s.\n\n\n')



    