import logging
import math
from selectors import EpollSelector
import sys
from argparse import Namespace
from ast import literal_eval
from pathlib import Path
from typing import Optional, Tuple, List
from timeit import default_timer as timer

import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils import data

sys.path.append(str(Path(__file__).resolve().parent.parent))

from art.prop import AndProp
from art.bisecter import Bisecter
from art import exp, acas, utils

RES_DIR = Path(__file__).resolve().parent.parent / 'results' / 'acas' / 'repair' / 'debug'
RES_DIR.mkdir(parents=True, exist_ok=True)
REPAIR_MODEL_DIR = Path(__file__).resolve().parent.parent / 'model' / 'reassure_format'
REPAIR_MODEL_DIR.mkdir(parents=True, exist_ok=True)


class AcasArgParser(exp.ExpArgParser):
    """ Parsing and storing all ACAS experiment configuration arguments. """

    def __init__(self, log_path: Optional[str], *args, **kwargs):
        super().__init__(log_path, *args, **kwargs)

        # use repair or nor
        self.add_argument('--no_repair', type=bool, default=True, 
                        help='not repair use incremental')
        
        # the combinational form of support and patch net
        self.add_argument('--reassure_support_and_patch_combine',type=bool, default=False,
                        help='use REASSURE method to combine support and patch network')

        # training
        self.add_argument('--accuracy_loss', type=str, choices=['L1', 'MSE', 'CE'], default='CE',
                          help='canonical loss function for concrete points training')
        self.add_argument('--sample_amount', type=int, default=5000,
                          help='specifically for data points sampling from spec')
        self.add_argument('--reset_params', type=literal_eval, default=False,
                          help='start with random weights or provided trained weights when available')

        # querying a verifier
        self.add_argument('--max_verifier_sec', type=int, default=300,
                          help='allowed time for a verifier query')
        self.add_argument('--verifier_timeout_as_safe', type=literal_eval, default=True,
                          help='when verifier query times out, treat it as verified rather than unknown')

        self.set_defaults(exp_fn='test_goal_safety', use_scheduler=True)
        return

    def setup_rest(self, args: Namespace):
        super().setup_rest(args)

        def ce_loss(outs: Tensor, labels: Tensor):
            softmax = nn.Softmax(dim=1)
            ce = nn.CrossEntropyLoss()
            # *= -1 because ACAS picks smallest value as suggestion
            return ce(softmax(outs * -1.), labels)

        args.accuracy_loss = {
            'L1': nn.L1Loss(),
            'MSE': nn.MSELoss(),
            'CE': ce_loss
        }[args.accuracy_loss]
        return
    pass


class AcasPoints(exp.ConcIns):
    """ Storing the concrete data points for one ACAS network sampled.
        Loads to CPU/GPU automatically.
    """
    @classmethod
    def load(cls, nid: acas.AcasNetID, train: bool, device):
        suffix = 'train' if train else 'test'
        fname = f'{str(nid)}-orig-{suffix}.pt'  # note that it is using original data
        combine = torch.load(Path(acas.ACAS_DIR, fname), device)
        inputs, labels = combine
        assert len(inputs) == len(labels)
        return cls(inputs, labels)
    pass


def eval_test(net: acas.AcasNet, testset: AcasPoints, categories=None) -> float:
    """ Evaluate accuracy on test set.
    :param categories: e.g., acas.AcasOut
    """
    with torch.no_grad():
        outs = net(testset.inputs) * -1
        predicted = outs.argmax(dim=1)
        correct = (predicted == testset.labels).sum().item()
        ratio = correct / len(testset)

        # per category
        if categories is not None:
            for cat in categories:
                idxs = testset.labels == cat
                cat_predicted = predicted[idxs]
                cat_labels = testset.labels[idxs]
                cat_correct = (cat_predicted == cat_labels).sum().item()
                cat_ratio = math.nan if len(cat_labels) == 0 else cat_correct / len(cat_labels)
                logging.debug(f'--For category {cat}, out of {len(cat_labels)} items, ratio {cat_ratio}')
    return ratio


from repair_moudle import SupportNet, PatchNet, IntersectionNetSum
def repair_acas(nid: acas.AcasNetID, args: Namespace, weight_clamp = False)-> Tuple[int, float, bool, float]:
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
    logging.info(repair_net)


    all_props = AndProp(nid.applicable_props(args.dom))
    v = Bisecter(args.dom, all_props)

    def run_abs(net, batch_abs_lb: Tensor, batch_abs_ub: Tensor, batch_abs_bitmap: Tensor) -> Tensor:
        """ Return the safety distances over abstract domain. """
        batch_abs_ins = args.dom.Ele.by_intvl(batch_abs_lb, batch_abs_ub)
        batch_abs_outs = net(batch_abs_ins)
        return all_props.safe_dist(batch_abs_outs, batch_abs_bitmap)
    
    

    
    in_lb, in_ub = all_props.lbub(device)
    in_bitmap = all_props.bitmap(device)
    in_lb = net.normalize_inputs(in_lb, bound_mins, bound_maxs)
    in_ub = net.normalize_inputs(in_ub, bound_mins, bound_maxs)

    # already moved to GPU if necessary
    trainset = AcasPoints.load(nid, train=True, device=device)
    testset = AcasPoints.load(nid, train=False, device=device)

    start = timer()

    if args.no_abs or args.no_refine:
        curr_abs_lb, curr_abs_ub, curr_abs_bitmap = in_lb, in_ub, in_bitmap
    else:
        # refine it at the very beginning to save some steps in later epochs
        # TODO 这里可以先用原网络的结构对抽象域进行细化，然后再修复
        curr_abs_lb, curr_abs_ub, curr_abs_bitmap = v.split(in_lb, in_ub, in_bitmap, net, args.refine_top_k,
                                                            # tiny_width=args.tiny_width,
                                                            stop_on_k_all=args.start_abs_cnt)

    # params = list(net.parameters()) 
    # for patch in patch_lists:
    #     params.extend(patch.parameters())
    # for support in support_lists:
    #     params.extend(support.parameters())
    

    
    if args.reassure_support_and_patch_combine:
        def get_out_abs(net, batch_abs_lb: Tensor, batch_abs_ub: Tensor)-> Tuple[Tensor]:
            """ Return the outputs of over abstract domain. """
            batch_abs_ins = args.dom.Ele.by_intvl(batch_abs_lb, batch_abs_ub)
            batch_abs_outs = net(batch_abs_ins)
            batch_abs_lb = batch_abs_outs.lb()
            batch_abs_ub = batch_abs_outs.ub()
            return batch_abs_lb, batch_abs_ub

        def choose_k(patch_lists: List[nn.Module]) -> List[Tensor] :
            K_list = []
            for patch in patch_lists:
                batch_abs_lb, batch_abs_ub = get_out_abs(patch, curr_abs_lb, curr_abs_ub)
                lb = torch.min(batch_abs_lb)
                ub = torch.max(batch_abs_ub)
                
                result = 0.1*torch.maximum(ub, torch.abs(lb)).item()
                K_list.append(result)
            return K_list
        k_list = choose_k(patch_lists=patch_lists)
        repair_net.K = k_list

    
    opti = Adam(repair_net.parameters(), lr=args.lr)
    scheduler = args.scheduler_fn(opti)  # could be None

    accuracies = []  # epoch 0: ratio
    certified = False
    epoch = 0

    while True:
        # first, evaluate current model
        logging.info(f'[{utils.time_since(start)}] After epoch {epoch}:')
        if not args.no_pts:
            logging.info(f'Loaded {trainset.real_len()} points for training.')

        if not args.no_abs:
            logging.info(f'Loaded {len(curr_abs_lb)} abstractions for training.')
            with torch.no_grad():
                full_dists = run_abs(repair_net,curr_abs_lb, curr_abs_ub, curr_abs_bitmap)
            logging.info(f'min loss {full_dists.min()}, max loss {full_dists.max()}.')
            if full_dists.max() <= 0:
                certified = True
                logging.info(f'All {len(curr_abs_lb)} abstractions certified.')
            else:
                _, worst_idx = full_dists.max(dim=0)
                logging.info(f'Max loss at LB: {curr_abs_lb[worst_idx]}, UB: {curr_abs_ub[worst_idx]}, rule: {curr_abs_bitmap[worst_idx]}.')

        accuracies.append(eval_test(repair_net, testset))
        logging.info(f'Test set accuracy {accuracies[-1]}.')

        # check termination
        if certified and epoch >= args.min_epochs:
            # all safe and sufficiently trained
            break

        if epoch >= args.max_epochs:
            break

        epoch += 1
        certified = False
        logging.info(f'\n[{utils.time_since(start)}] Starting epoch {epoch}:')

        absset = exp.AbsIns(curr_abs_lb, curr_abs_ub, curr_abs_bitmap)

        # dataset may have expanded, need to update claimed length to date
        if not args.no_pts:
            trainset.reset_claimed_len()
        if not args.no_abs:
            absset.reset_claimed_len()
        if (not args.no_pts) and (not args.no_abs):
            ''' Might simplify this to just using the amount of abstractions, is it unnecessarily complicated? '''
            # need to enumerate both
            max_claimed_len = max(trainset.claimed_len, absset.claimed_len)
            trainset.claimed_len = max_claimed_len
            absset.claimed_len = max_claimed_len

        if not args.no_pts:
            conc_loader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
            nbatches = len(conc_loader)
            conc_loader = iter(conc_loader)
        if not args.no_abs:
            abs_loader = data.DataLoader(absset, batch_size=args.batch_size, shuffle=True)
            nbatches = len(abs_loader)  # doesn't matter rewriting len(conc_loader), they are the same
            abs_loader = iter(abs_loader)

        total_loss = 0.
        for i in range(nbatches):
            opti.zero_grad()
            batch_loss = 0.
            if not args.no_pts:
                batch_inputs, batch_labels = next(conc_loader)
                batch_outputs = repair_net(batch_inputs)
                batch_loss += args.accuracy_loss(batch_outputs, batch_labels)
            if not args.no_abs:
                batch_abs_lb, batch_abs_ub, batch_abs_bitmap = next(abs_loader)
                batch_dists = run_abs(repair_net,batch_abs_lb, batch_abs_ub, batch_abs_bitmap)
                
                #这里又对batch的loss求了个均值，作为最后的safe_loss(下面的没看懂，好像类似于l1)
                safe_loss = batch_dists.mean()  # L1, need to upgrade to batch_worsts to unlock loss other than L1
                total_loss += safe_loss.item()
                batch_loss += safe_loss
            logging.debug(f'Epoch {epoch}: {i / nbatches * 100 :.2f}%. Batch loss {batch_loss.item()}')
            batch_loss.backward()
            opti.step()


        
        total_loss /= nbatches

        # 修改学习率
        if scheduler is not None:
            scheduler.step(total_loss)
        logging.info(f'[{utils.time_since(start)}] At epoch {epoch}: avg accuracy training loss {total_loss}.')

        # Refine abstractions, note that restart from scratch may output much fewer abstractions thus imprecise.
        # TODO 这里继承在net上refine的输入，
        if (not args.no_refine) and len(curr_abs_lb) < args.max_abs_cnt:
            curr_abs_lb, curr_abs_ub, curr_abs_bitmap = v.split(curr_abs_lb, curr_abs_ub, curr_abs_bitmap, repair_net,
                                                                args.refine_top_k,
                                                                # tiny_width=args.tiny_width,
                                                                stop_on_k_new=args.refine_top_k)
        pass
    train_time = timer() - start
    logging.info(f'Accuracy at every epoch: {accuracies}')
    logging.info(f'After {epoch} epochs / {utils.pp_time(train_time)}, ' +
                 f'eventually the trained network got certified? {certified}, ' +
                 f'with {accuracies[-1]:.4f} accuracy on test set.')
    # torch.save(repair_net.state_dict(), str(REPAIR_MODEL_DIR / '2_9.pt'))
    # net.save_nnet(f'./ART_{nid.x.numpy().tolist()}_{nid.y.numpy().tolist()}_repair2p_noclamp_epoch_{epoch}.nnet',
    #             mins = bound_mins, maxs = bound_maxs) 
    
    return epoch, train_time, certified, accuracies[-1]
    pass

def train_acas(nid: acas.AcasNetID, args: Namespace, weight_clamp = False) -> Tuple[int, float, bool, float]:
    """ The almost completed skeleton of training ACAS networks using ART.
    :return: trained_epochs, train_time, certified, final accuracies
    """
    fpath = nid.fpath()
    net, bound_mins, bound_maxs = acas.AcasNet.load_nnet(fpath, args.dom, device)
    if args.reset_params:
        net.reset_parameters()
    logging.info(net)

    all_props = AndProp(nid.applicable_props(args.dom))
    v = Bisecter(args.dom, all_props)

    def run_abs(batch_abs_lb: Tensor, batch_abs_ub: Tensor, batch_abs_bitmap: Tensor) -> Tensor:
        """ Return the safety distances over abstract domain. """
        batch_abs_ins = args.dom.Ele.by_intvl(batch_abs_lb, batch_abs_ub)
        batch_abs_outs = net(batch_abs_ins)
        return all_props.safe_dist(batch_abs_outs, batch_abs_bitmap)

    # 包含所有的性质(以及若两个性质的输入区间有交，则将交集单独拿出来，
    # 并设置其对应的safe_dist function为两个性质的safe_dist相加)
    in_lb, in_ub = all_props.lbub(device)
    in_bitmap = all_props.bitmap(device)
    in_lb = net.normalize_inputs(in_lb, bound_mins, bound_maxs)
    in_ub = net.normalize_inputs(in_ub, bound_mins, bound_maxs)

    # already moved to GPU if necessary
    trainset = AcasPoints.load(nid, train=True, device=device)
    testset = AcasPoints.load(nid, train=False, device=device)

    start = timer()

    if args.no_abs or args.no_refine:
        curr_abs_lb, curr_abs_ub, curr_abs_bitmap = in_lb, in_ub, in_bitmap
    else:
        # refine it at the very beginning to save some steps in later epochs
        curr_abs_lb, curr_abs_ub, curr_abs_bitmap = v.split(in_lb, in_ub, in_bitmap, net, args.refine_top_k,
                                                            # tiny_width=args.tiny_width,
                                                            stop_on_k_all=args.start_abs_cnt)

    opti = Adam(net.parameters(), lr=args.lr)
    scheduler = args.scheduler_fn(opti)  # could be None
    origin_parameters_list = []
    # for origin_i_parameter in net.parameters():
    #     origin_parameters_list.append(origin_i_parameter.data.clone())

    accuracies = []  # epoch 0: ratio
    certified = False
    epoch = 0
    record_full_dists_list = [100]
    while True:
        # first, evaluate current model
        logging.info(f'[{utils.time_since(start)}] After epoch {epoch}:')
        if not args.no_pts:
            logging.info(f'Loaded {trainset.real_len()} points for training.')
        # if epoch == 44:
        #     print('full_dists:\n',full_dists)
        #     pass
        if not args.no_abs:
            logging.info(f'Loaded {len(curr_abs_lb)} abstractions for training.')
            with torch.no_grad():
                full_dists = run_abs(curr_abs_lb, curr_abs_ub, curr_abs_bitmap)
            logging.info(f'min loss {full_dists.min()}, max loss {full_dists.max()}.')
            # print("min_full_dist_max_history:",min(record_full_dists_list))
            # if full_dists.max() < min(record_full_dists_list):
            #     s = full_dists.max().cpu().numpy()
            #     net.save_nnet(f'./0.01/ART_{nid.x.numpy().tolist()}_{nid.y.numpy().tolist()}_repair2p_clamp_0.01_epoch_{epoch}_ldloss_max_{s}.nnet',
            #     mins = bound_mins, maxs = bound_maxs) 
            # print("full_dist_max_now",full_dists.max())
            # record_full_dists_list.append(full_dists.max())
            if full_dists.max() <= 0:
                certified = True
                logging.info(f'All {len(curr_abs_lb)} abstractions certified.')
            else:
                _, worst_idx = full_dists.max(dim=0)
                logging.info(f'Max loss at LB: {curr_abs_lb[worst_idx]}, UB: {curr_abs_ub[worst_idx]}, rule: {curr_abs_bitmap[worst_idx]}.')

        accuracies.append(eval_test(net, testset))
        logging.info(f'Test set accuracy {accuracies[-1]}.')

        # check termination
        if certified and epoch >= args.min_epochs:
            # all safe and sufficiently trained
            break

        if epoch >= args.max_epochs:
            break

        epoch += 1
        certified = False
        logging.info(f'\n[{utils.time_since(start)}] Starting epoch {epoch}:')

        absset = exp.AbsIns(curr_abs_lb, curr_abs_ub, curr_abs_bitmap)

        # dataset may have expanded, need to update claimed length to date
        if not args.no_pts:
            trainset.reset_claimed_len()
        if not args.no_abs:
            absset.reset_claimed_len()
        if (not args.no_pts) and (not args.no_abs):
            ''' Might simplify this to just using the amount of abstractions, is it unnecessarily complicated? '''
            # need to enumerate both
            max_claimed_len = max(trainset.claimed_len, absset.claimed_len)
            trainset.claimed_len = max_claimed_len
            absset.claimed_len = max_claimed_len

        if not args.no_pts:
            conc_loader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
            nbatches = len(conc_loader)
            conc_loader = iter(conc_loader)
        if not args.no_abs:
            abs_loader = data.DataLoader(absset, batch_size=args.batch_size, shuffle=True)
            nbatches = len(abs_loader)  # doesn't matter rewriting len(conc_loader), they are the same
            abs_loader = iter(abs_loader)

        total_loss = 0.
        for i in range(nbatches):
            opti.zero_grad()
            batch_loss = 0.
            if not args.no_pts:
                batch_inputs, batch_labels = next(conc_loader)
                batch_outputs = net(batch_inputs)
                batch_loss += args.accuracy_loss(batch_outputs, batch_labels)
            if not args.no_abs:
                batch_abs_lb, batch_abs_ub, batch_abs_bitmap = next(abs_loader)
                batch_dists = run_abs(batch_abs_lb, batch_abs_ub, batch_abs_bitmap)
                
                #这里又对batch的loss求了个均值，作为最后的safe_loss(下面的没看懂，好像类似于l1)
                safe_loss = batch_dists.mean()  # L1, need to upgrade to batch_worsts to unlock loss other than L1
                total_loss += safe_loss.item()
                batch_loss += safe_loss
            logging.debug(f'Epoch {epoch}: {i / nbatches * 100 :.2f}%. Batch loss {batch_loss.item()}')
            batch_loss.backward()
            # now_parameters_list = []
            # for now_i_parameter in net.parameters():
            #     now_parameters_list.append(now_i_parameter.data.clone())

            def get_clamp_weights(grad,weight_top,weight_below):
                for i, combine_i in enumerate(zip(weight_below,weight_top)):
                    (below_i,top_i) = combine_i
                    for j,combine_j in enumerate(zip(below_i,top_i)):
                        below_ij,top_ij = combine_j
                        min = torch.min(below_ij,top_ij)
                        max = torch.max(below_ij,top_ij)
                        if grad[i][j] < min:
                            grad[i][j] = min
                        elif grad[i][j] > max:
                            grad[i][j] = max
                        else:
                            continue
                pass
                return grad
            
            def get_clamp_bias(grad,bias_top,bias_below):
                for i,combine_i in enumerate(zip(bias_below,bias_top)):
                    below_ij,top_ij = combine_i
                    min = torch.min(below_ij,top_ij)
                    max = torch.max(below_ij,top_ij)
                    if grad[i] < min:
                        grad[i] = min
                    elif grad[i] > max:
                        grad[i] = max
                    else:
                        continue
                pass
                return grad
            opti.step()
            if weight_clamp:
                with torch.no_grad():
                    for i,parameter in enumerate(net.parameters()):
                        parameter_i_min = origin_parameters_list[i]*(0.99)
                        parameter_i_max = origin_parameters_list[i]*(1.01)
                        newp = parameter.clamp(min = parameter_i_min,max=parameter_i_max)
                        parameter.data = newp
                        pass
            # print(parameter.requires_grad)

        # inspect the trained weights after another epoch
        # meta.inspect_params(net.state_dict())

        total_loss /= nbatches
        if scheduler is not None:
            scheduler.step(total_loss)
        logging.info(f'[{utils.time_since(start)}] At epoch {epoch}: avg accuracy training loss {total_loss}.')

        # Refine abstractions, note that restart from scratch may output much fewer abstractions thus imprecise.
        if (not args.no_refine) and len(curr_abs_lb) < args.max_abs_cnt:
            curr_abs_lb, curr_abs_ub, curr_abs_bitmap = v.split(curr_abs_lb, curr_abs_ub, curr_abs_bitmap, net,
                                                                args.refine_top_k,
                                                                # tiny_width=args.tiny_width,
                                                                stop_on_k_new=args.refine_top_k)
        pass

    # summarize
    train_time = timer() - start
    logging.info(f'Accuracy at every epoch: {accuracies}')
    logging.info(f'After {epoch} epochs / {utils.pp_time(train_time)}, ' +
                 f'eventually the trained network got certified? {certified}, ' +
                 f'with {accuracies[-1]:.4f} accuracy on test set.')
    # net.save_nnet(f'./ART_{nid.x.numpy().tolist()}_{nid.y.numpy().tolist()}_repair2p_noclamp_epoch_{epoch}.nnet',
    #             mins = bound_mins, maxs = bound_maxs) 
    
    return epoch, train_time, certified, accuracies[-1]



def _run(nids: List[acas.AcasNetID], args: Namespace):
    """ Run for different networks with specific configuration. """
    res = []
    for nid in nids:
        logging.info(f'For {nid}')
        outs = train_acas(nid, args,weight_clamp=False)
        res.append(outs)

    avg_res = torch.tensor(res).mean(dim=0)
    logging.info(f'=== Avg <epochs, train_time, certified, accuracy> for {len(nids)} networks:')
    logging.info(avg_res)
    return

def _run_repair(nids: List[acas.AcasNetID], args: Namespace):
    """ Run for different networks with specific configuration. """
    logging.info('===== start repair ======')
    res = []
    for nid in nids:
        logging.info(f'For {nid}')
        outs = repair_acas(nid, args,weight_clamp=False)
        res.append(outs)

    avg_res = torch.tensor(res).mean(dim=0)
    logging.info(f'=== Avg <epochs, train_time, certified, accuracy> for {len(nids)} networks:')
    logging.info(avg_res)
    return




def test_goal_safety(parser: AcasArgParser):
    """ Q1: Show that we can train previously unsafe networks to safe. """
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
    if args.no_repair:
        _run(nids, args)
    else:
        _run_repair(nids, args)
    return

def test_goal_adv(parser: AcasArgParser):
    """ Q1: Show that we can train previously unsafe networks to safe. """
    defaults = {
        # 'start_abs_cnt': 5000,
        'batch_size': 100,  # to make it faster
        'min_epochs': 25,
        'max_epochs': 35
    }
    parser.set_defaults(**defaults)
    args = parser.parse_args()

    logging.info(utils.fmt_args(args))
    nids = acas.AcasNetID.goal_adv_ids(args.dom)
    _run(nids, args)
    return


def test_goal_accuracy(parser: AcasArgParser):
    """ Q2: Show that the safe-by-construction overhead on accuracy is mild. """
    defaults = {
        # 'start_abs_cnt': 5000,
        'batch_size': 100,  # to make it faster
        'min_epochs': 25,
        'max_epochs': 35
    }
    parser.set_defaults(**defaults)
    args = parser.parse_args()

    logging.info(utils.fmt_args(args))
    nids = acas.AcasNetID.goal_accuracy_ids(args.dom)
    _run(nids, args)
    return






if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    test_defaults = {
        'exp_fn': 'test_goal_safety',
        # 'exp_fn': 'test_patch_distribution',

        'no_repair': True,
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
