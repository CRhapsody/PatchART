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

from art.prop import AndProp,FeatureAndProp
from art.bisecter import Bisecter
from art import exp, utils

from mnist import MnistNet, MnistFeatureProp

MNIST_DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'MNIST' / 'processed'
MNIST_NET_DIR = Path(__file__).resolve().parent.parent / 'pgd' / 'model'
RES_DIR = Path(__file__).resolve().parent.parent / 'results' / 'mnist' / 'repair' / 'debug'
RES_DIR.mkdir(parents=True, exist_ok=True)
REPAIR_MODEL_DIR = Path(__file__).resolve().parent.parent / 'model' / 'reassure_format'
REPAIR_MODEL_DIR.mkdir(parents=True, exist_ok=True)


class MnistArgParser(exp.ExpArgParser):
    """ Parsing and storing all ACAS experiment configuration arguments. """

    def __init__(self, log_path: Optional[str], *args, **kwargs):
        super().__init__(log_path, *args, **kwargs)

        # use repair or nor
        self.add_argument('--no_repair', type=bool, default=True, 
                        help='not repair use incremental')
        
        # the combinational form of support and patch net
        self.add_argument('--reassure_support_and_patch_combine',type=bool, default=False,
                        help='use REASSURE method to combine support and patch network')

        self.add_argument('--repair_radius',type=float, default=0.2, 
                          help='the radius of repairing datas or features')

        # training
        self.add_argument('--accuracy_loss', type=str, choices=['L1', 'MSE', 'CE'], default='CE',
                          help='canonical loss function for concrete points training')
        self.add_argument('--support_loss', type=str, choices=['L1', 'MSE', 'CE','BCE'], default='BCE',
                          help= 'canonical loss function for patch net training')
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

        def Bce_loss(outs: Tensor, labels: Tensor):
            bce = nn.BCEWithLogitsLoss()
            return bce(outs, labels)

        args.accuracy_loss = {
            'L1': nn.L1Loss(),
            'MSE': nn.MSELoss(),
            'CE': ce_loss
        }[args.accuracy_loss]

        args.support_loss = {
            'BCE' : Bce_loss
        }[args.support_loss]
        return
    pass

class MnistPoints(exp.ConcIns):
    """ Storing the concrete data points for one ACAS network sampled.
        Loads to CPU/GPU automatically.
    """
    @classmethod
    def load(cls, train: bool, device):
        suffix = 'train' if train else 'test'
        fname = f'{suffix}_attack_data_part.pt'  # note that it is using original data
        combine = torch.load(Path(MNIST_DATA_DIR, fname), device)
        inputs, labels = combine
        assert len(inputs) == len(labels)
        return cls(inputs, labels)
    pass

def eval_test(net: MnistNet, testset: MnistPoints, categories=None) -> float:
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


from repair_moudle import SupportNet, PatchNet, NetFeatureSum,Netsum
def repair_mnist(args: Namespace, weight_clamp = False)-> Tuple[int, float, bool, float]:
    fname = 'pdg_net.pth'
    fpath = Path(MNIST_NET_DIR, fname)


    trainset = MnistPoints.load(train=True, device=device)
    testset = MnistPoints.load(train=False, device=device)

    net = MnistNet(dom=args.dom)
    net.to(device)
    net.load_state_dict(torch.load(fpath, map_location=device))


    # bound_mins = torch.zeros_like(trainset[0])
    # bound_maxs = torch.ones_like(trainset[0])

    # TODO clusttering

    # get the features of the dataset using mnist_net.split
    model1, model2 = net.split()
    feature = model1(trainset.inputs)

    feature_trainset = MnistPoints(feature, trainset.labels)


    # set the box bound of the features, which the length of the box is 2*radius
    # bound_mins = feature - radius
    # bound_maxs = feature + radius

    # the steps are as follows:
    # 1. TODO construct a mnist prop file include Mnistprop Module, which inherits from Oneprop
    # 2. TODO use the features to construct the Mnistprop
    # 3. TODO use the Mnistprop to construct the Andprop(join) 

    featurelist = [(data[0],data[1]) for data in zip(feature, trainset.labels)]
    input_size = feature.size()[1]

    feature_prop_list = MnistFeatureProp.all_props(args.dom, DataList=featurelist, input_dimension = input_size,radius= args.repair_radius)

    # get the all props after join all l_0 ball feature property
    all_props = FeatureAndProp(props=feature_prop_list)
    v = Bisecter(args.dom, all_props)

    in_lb, in_ub = all_props.lbub(device)
    in_bitmap = all_props.bitmap(device)

    
    # repair part
    # the number of repair patch network,which is equal to the number of properties
    n_repair = MnistFeatureProp.LABEL_NUMBER
    # the construction of support and patch network
    
    hidden_size = [10,10,10,10]
    patch_lists = []

    support_net = SupportNet(input_size=input_size, dom=args.dom, hidden_sizes=hidden_size, output_size=n_repair,
            name = f'support network')
    support_net.to(device)
    
    for i in range(n_repair):
        patch_net = PatchNet(input_size=input_size, dom=args.dom, hidden_sizes=hidden_size, output_size=n_repair,
            name = f'patch network {i}')
        patch_net.to(device)
        patch_lists.append(patch_net)



    def run_abs(net, batch_abs_lb: Tensor, batch_abs_ub: Tensor, batch_abs_bitmap: Tensor) -> Tensor:
        """ Return the safety distances over abstract domain. """
        batch_abs_ins = args.dom.Ele.by_intvl(batch_abs_lb, batch_abs_ub)
        batch_abs_outs = net(batch_abs_ins)
        return all_props.safe_feature_dist(batch_abs_outs, batch_abs_bitmap)
    


    # train the support network
    # def train_model(model, safe_lb, safe_ub, safe_extra, wl_lb, wl_ub, wl_extra):
    def train_model(support_net, in_lb, in_ub, in_bitmap):
        # training support network using the splited bounds
        opti_support = Adam(support_net.parameters(), lr=args.lr)
        scheduler_support = args.scheduler_fn(opti_support)  # could be None

        # certain epoch to train support network
        initial_training_support_epoch = 400

        criterion = args.support_loss  # 分类任务的损失函数

        # binary_criterion = args.support_loss  # 正反例识别任务的损失函数

        # input_region_dataloader = process_data(safe_lb, safe_ub, safe_extra, wl_lb, wl_ub, wl_extra)

        # construct the abstract dataset for support network training
        # absset = exp.AbsIns(in_lb, in_ub, in_bitmap)
        # abs_loader = data.DataLoader(absset, batch_size=len(in_lb), shuffle=True)
        # nbatches = len(abs_loader)  # doesn't matter rewriting len(conc_loader), they are the same
        # abs_loader = iter(abs_loader)[0]
        logging.info('start pre-training support network:')
        with torch.enable_grad():
            for epoch in range(initial_training_support_epoch):
                # for j in range(len(nbatches)):
                loss = 0
                opti_support.zero_grad()
                # train_data, class_labels, binary_labels = next(input_region_dataloader)
                # class_output, binary_output = support_net(train_data)
                # train_data, class_labels = next(abs_loader)
                # class_output = support_net(train_data)
                # loss = criterion(class_output, class_labels)
                # binary_loss = binary_criterion(binary_output, binary_labels)
                
                # TODO (complete) rewrite the safe dists
                # TODO (complete) write a abstract domain of sigmoid function
                # batch_dists = run_abs(support_net, in_lb, in_ub, in_bitmap)
                abs_ins = args.dom.Ele.by_intvl(in_lb, in_ub)
                abs_outs = support_net(abs_ins)
                loss = all_props.safe_feature_dist(abs_outs, in_bitmap)

                # TODO can adjust
                loss = loss.mean()

                # only one property
                if loss is None:
                    break

                # TODO as above(complete, not batch) 
                # loss = batch_dists.mean()
                

                
                # loss = class_loss + binary_loss
                loss.backward()
                opti_support.step()
                if epoch % 10 == 0:
                    logging.info(f'Epoch {epoch}: support loss {loss.item()}')
                if scheduler_support is not None:
                    scheduler_support.step(loss)


    # TODO it is necessary to test the support network?

    #first train the support network
    with torch.no_grad():
        train_model(support_net, in_lb, in_ub, in_bitmap)


    # TODO(complete) construct the repair network 
    # the number of repair patch network,which is equal to the number of properties 
    finally_net =  NetFeatureSum(args.dom, target_net = net, support_nets= support_net, patch_nets= patch_lists, device=device)


    start = timer()

    if args.no_abs or args.no_refine:
        curr_abs_lb, curr_abs_ub, curr_abs_bitmap = in_lb, in_ub, in_bitmap
    else:
        # refine it at the very beginning to save some steps in later epochs
        # and use the refined bounds as the initial bounds for support network training
        
        curr_abs_lb, curr_abs_ub, curr_abs_bitmap = v.split(in_lb, in_ub, in_bitmap, model2, args.refine_top_k,
                                                                tiny_width=args.tiny_width,
                                                                stop_on_k_all=args.start_abs_cnt,for_feature=True) #,for_support=False
    
    # repair the classifer without feature extractor
    repair_net = Netsum(args.dom, target_net = model2, support_nets= support_net, patch_nets= patch_lists, device=device)

    # train the patch and original network
    opti = Adam(repair_net.parameters(), lr=args.lr)
    scheduler = args.scheduler_fn(opti)  # could be None

    # freeze the parameters of the original network for extracting features
    for name, param in repair_net.named_parameters():
        if 'conv1' or 'conv2' or 'fc1' in name:
            param.requires_grad = False
            opti.param_groups[0]['params'].append(param)
            opti.param_groups[0]['lr'] = 0

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

        # test the repaired model which combines the feature extractor, classifier and the patch network
        accuracies.append(eval_test(finally_net, testset))
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
            feature_trainset.reset_claimed_len()
        if not args.no_abs:
            absset.reset_claimed_len()
        if (not args.no_pts) and (not args.no_abs):
            ''' Might simplify this to just using the amount of abstractions, is it unnecessarily complicated? '''
            # need to enumerate both
            max_claimed_len = max(feature_trainset.claimed_len, absset.claimed_len)
            feature_trainset.claimed_len = max_claimed_len
            absset.claimed_len = max_claimed_len

        if not args.no_pts:
            conc_loader = data.DataLoader(feature_trainset, batch_size=args.batch_size, shuffle=True)
            nbatches = len(conc_loader)
            conc_loader = iter(conc_loader)
        if not args.no_abs:
            abs_loader = data.DataLoader(absset, batch_size=args.batch_size, shuffle=True)
            nbatches = len(abs_loader)  # doesn't matter rewriting len(conc_loader), they are the same
            abs_loader = iter(abs_loader)

        total_loss = 0.
        with torch.enable_grad():
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

                #TODO 这里怎么办
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
                                                                stop_on_k_new=args.refine_top_k,for_feature=True)
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




# def _run(nids: List[acas.AcasNetID], args: Namespace):
#     """ Run for different networks with specific configuration. """
#     res = []
#     for nid in nids:
#         logging.info(f'For {nid}')
#         outs = train_acas(nid, args,weight_clamp=False)
#         res.append(outs)

#     avg_res = torch.tensor(res).mean(dim=0)
#     logging.info(f'=== Avg <epochs, train_time, certified, accuracy> for {len(nids)} networks:')
#     logging.info(avg_res)
#     return

def _run_repair(args: Namespace):
    """ Run for different networks with specific configuration. """
    logging.info('===== start repair ======')
    res = []
    # for nid in nids:
    logging.info(f'For pgd attack net')
    outs = repair_mnist(args,weight_clamp=False)
    res.append(outs)

    avg_res = torch.tensor(res).mean(dim=0)
    logging.info(f'=== Avg <epochs, train_time, certified, accuracy> for pgd attack networks:')
    logging.info(avg_res)
    return




def test_goal_safety(parser: MnistArgParser):
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
    # nids = acas.AcasNetID.goal_safety_ids(args.dom)
    if args.no_repair:
        print('why not repair?')
    else:
        _run_repair(args)
    return

# def test_goal_adv(parser: MnistArgParser):
#     """ Q1: Show that we can train previously unsafe networks to safe. """
#     defaults = {
#         # 'start_abs_cnt': 5000,
#         'batch_size': 100,  # to make it faster
#         'min_epochs': 25,
#         'max_epochs': 35
#     }
#     parser.set_defaults(**defaults)
#     args = parser.parse_args()

#     logging.info(utils.fmt_args(args))
#     nids = acas.AcasNetID.goal_adv_ids(args.dom)
#     _run(nids, args)
#     return


# def test_goal_accuracy(parser: MnistArgParser):
#     """ Q2: Show that the safe-by-construction overhead on accuracy is mild. """
#     defaults = {
#         # 'start_abs_cnt': 5000,
#         'batch_size': 100,  # to make it faster
#         'min_epochs': 25,
#         'max_epochs': 35
#     }
#     parser.set_defaults(**defaults)
#     args = parser.parse_args()

#     logging.info(utils.fmt_args(args))
#     nids = acas.AcasNetID.goal_accuracy_ids(args.dom)
#     _run(nids, args)
#     return






if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    test_defaults = {
        'exp_fn': 'test_goal_safety',
        # 'exp_fn': 'test_patch_distribution',
        # 'no_refine': True
        'no_repair': False,
        'reassure_support_and_patch_combine': True,
        'repair_radius': 0.1,

        
    }
    parser = MnistArgParser(RES_DIR, description='MNIST Correct by Construction')
    parser.set_defaults(**test_defaults)
    args, _ = parser.parse_known_args()

    exp_fn = locals()[args.exp_fn]
    start = timer()
    exp_fn(parser)

    logging.info(f'Total Cost Time: {timer() - start}s.\n\n\n')
    pass
