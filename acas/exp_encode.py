# property 3 need coc is not min

# if coc is min, then the output is the second smallest
# need coc to be infty

import logging
import math
from selectors import EpollSelector
import sys
from argparse import Namespace
from ast import literal_eval
from pathlib import Path
from typing import Optional, Tuple, List
from timeit import default_timer as timer
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils import data
import acas_utils as acas


from art.prop import AndProp
from art.bisecter import Bisecter
from art import exp, utils
from DiffAbs.DiffAbs import deeppoly
from art.repair_moudle import PatchNet, Netsum

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REPAIR_MODEL_DIR = Path(__file__).resolve().parent.parent / 'model' / 'acasxu'


class AcasPoints(exp.ConcIns):
    """ Storing the concrete data points for one ACAS network sampled.
        Loads to CPU/GPU automatically.
    """
    @classmethod
    def load(cls, nid: acas.AcasNetID, train: bool, device):
        suffix = 'train' if train else 'test'
        fname = f'{str(nid)}orig_{suffix}.pt'  # note that it is using original data
        combine = torch.load(Path(acas.ACAS_DIR, fname), device)
        inputs, labels = combine
        assert len(inputs) == len(labels)
        return cls(inputs, labels)
    pass

def get_bitmap(in_lb: Tensor, in_ub: Tensor, in_bitmap: Tensor, batch_inputs: Tensor,device = 'cuda'):
    '''
    in_lb: n_prop * input
    in_ub: n_prop * input
    batch_inputs: batch * input
    '''
    with torch.no_grad():
    
        batch_inputs_clone = batch_inputs.clone().unsqueeze_(1)
        # distingush the photo and the property
        if len(in_lb.shape) == 2:
            batch_inputs_clone = batch_inputs_clone.expand(batch_inputs.shape[0], in_lb.shape[0], in_lb.shape[1])
            is_in = (batch_inputs_clone >= in_lb) & (batch_inputs_clone <= in_ub)
            is_in = is_in.all(dim=-1) # every input is in the region of property, batch * n_prop
        elif len(in_lb.shape) == 4:
            if in_lb.shape[0] > 600:
                is_in_list = []
                for i in range(batch_inputs_clone.shape[0]):
                    batch_inputs_compare_datai = batch_inputs_clone[i].clone().expand(in_lb.shape[0], in_lb.shape[1], in_lb.shape[2], in_lb.shape[3])
                    is_in_datai = (batch_inputs_compare_datai >= in_lb) & (batch_inputs_compare_datai <= in_ub)
                    is_in_datai = is_in_datai.all(dim=(-1)).all(dim=(-1)).all(dim=(-1)) # every input is in the region of property, batch * n_prop
                    is_in_list.append(is_in_datai)
                is_in = torch.stack(is_in_list, dim=0)
            else:
                batch_inputs_clone = batch_inputs_clone.expand(batch_inputs.shape[0], in_lb.shape[0], in_lb.shape[1], in_lb.shape[2], in_lb.shape[3])
                is_in = (batch_inputs_clone >= in_lb) & (batch_inputs_clone <= in_ub)
                is_in = is_in.all(dim=(-1)).all(dim=(-1)).all(dim=(-1)) # every input is in the region of property, batch * n_prop
        # convert to bitmap
        bitmap = torch.zeros((batch_inputs.shape[0], in_bitmap.shape[1]), device = device).to(torch.uint8)
        bitmap_i, inbitmap_j =  is_in.nonzero(as_tuple=True)
        if bitmap_i.shape[0] != 0:
            bitmap[bitmap_i, :] = in_bitmap[inbitmap_j, :]
        else:
            pass


        return bitmap

# nids  = [acas.AcasNetID(torch.tensor(2),torch.tensor(1)),acas.AcasNetID(torch.tensor(1),torch.tensor(8)),acas.AcasNetID(torch.tensor(1),torch.tensor(9))]
nids = acas.AcasNetID.goal_safety_ids(dom=deeppoly)
nids.remove(acas.AcasNetID(torch.tensor(4),torch.tensor(2)))
print('len(nids):',len(nids))
acc_full = []
acc_full_repair = []
for nid in nids:
    # nid = nids[0]
    fpath  = nid.fpath()
    net, bound_mins, bound_maxs = acas.AcasNet.load_nnet(fpath, deeppoly, device)
    x = nid.x
    y = nid.y
    testset = AcasPoints.load(nid, False, device)

    # testset = torch.load(f'/home/chizm/PatchART/data/acas/acasxu_data_gene/n{x}{y}_counterexample_test.pt', device)
    test_data = testset.inputs

    # get the label
    with torch.no_grad():
        outs = net(test_data) 
        # outs.squeeze_(1)
        # let 0 index is infty negative
        predicted = outs.argmin(dim=1)
        test_label = predicted

    # load property
    all_props = AndProp(nid.applicable_props(deeppoly))
    in_lb, in_ub = all_props.lbub(device)
    in_bitmap = all_props.bitmap(device)
    in_lb = net.normalize_inputs(in_lb, bound_mins, bound_maxs)
    in_ub = net.normalize_inputs(in_ub, bound_mins, bound_maxs)

    bitmap = get_bitmap(in_lb, in_ub, in_bitmap, test_data, device)

    # encode
    with torch.no_grad():
        outs = net(test_data) 
        outs.squeeze_(1)
        # find the runner up index for all entries
        runner_up = outs.argsort(dim=1)[:,1]
        # let 0 index is infty negative
        # TODO 1: min
        # outs[bitmap[:,0] == 1,0] = torch.tensor(-math.inf)
        outs[:,0] = torch.tensor(-math.inf)
        # TODO 3: coc, weak-left is the min
        # outs[bitmap[:,0] == 1,1] = torch.tensor(-math.inf)
        # TODO 2:runner up
        # outs[bitmap[:,0] == 1,0] = outs[bitmap[:,0] == 1,runner_up[bitmap[:,0] == 1]]
        # 根据index of runner up，将runner up的值赋给 0 index
        # outs[:,0] = outs[torch.arange(outs.shape[0]),runner_up]
        predicted = outs.argmin(dim=1)
        correct = (predicted == test_label).sum().item()
        ratio = correct / len(testset)
        print(f'{nid} Accuracy: {correct}/{len(testset)} ({ratio:.2%})')
        acc_full.append(ratio)



    # n_repair = len(in_bitmap[0])
    # input_size = 5
    # hidden_size = [10,10,10]
    # patch_lists = []
    # for i in range(n_repair):
    #     patch_net = PatchNet(input_size=input_size, dom=deeppoly, hidden_sizes=hidden_size, output_size=5,
    #         name = f'patch network {i}')
    #     patch_net.to(device)
    #     patch_lists.append(patch_net)
    
    # repair_net = Netsum(deeppoly, target_net= net, patch_nets= patch_lists, device=device)
    # repair_net.load_state_dict(torch.load(REPAIR_MODEL_DIR / f'{str(nid)}repaired.pt'))
    # repair_net.eval()
    repair_net = torch.load(REPAIR_MODEL_DIR / f'{str(nid)}repaired.pt',map_location=device)
    with torch.no_grad():
        outs = repair_net(test_data, bitmap) * -1
        # outs.squeeze_(1)
        predicted = outs.argmax(dim=1)
        correct = (predicted == test_label).sum().item()
        ratio = correct / len(testset)
        print(f'repair {nid} Accuracy: {correct}/{len(testset)} ({ratio:.2%})')
        acc_full_repair.append(ratio)
print('acc_full:',sum(acc_full)/len(nids))
print('acc_full_repair:',sum(acc_full_repair)/len(nids))
