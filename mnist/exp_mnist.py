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
from art.prop import AndProp
from art.bisecter import Bisecter
from art import exp, utils
from art.repair_moudle import Netsum
from mnist.mnist_utils import MnistNet, MnistProp, Mnist_patch_model
# from mnist.u import MnistNet, MnistFeatureProp

MNIST_DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'MNIST' / 'processed'
MNIST_NET_DIR = Path(__file__).resolve().parent.parent / 'model' /'mnist'
RES_DIR = Path(__file__).resolve().parent.parent / 'results' / 'mnist' / 'repair' / 'debug'
RES_DIR.mkdir(parents=True, exist_ok=True)
REPAIR_MODEL_DIR = Path(__file__).resolve().parent.parent / 'model' / 'patch_format'
REPAIR_MODEL_DIR.mkdir(parents=True, exist_ok=True)


class MnistArgParser(exp.ExpArgParser):
    """ Parsing and storing all ACAS experiment configuration arguments. """

    def __init__(self, log_path: Optional[str], *args, **kwargs):
        super().__init__(log_path, *args, **kwargs)

        # use repair or nor
        self.add_argument('--no_repair', type=bool, default=True, 
                        help='not repair use incremental')
        self.add_argument('--repair_number', type=int, default=50,
                          help='the number of repair datas')
        self.add_argument('--repair_batchsize', type=int, default=1,
                            help='the batchsize of repair datas')
        
        # the combinational form of support and patch net
        self.add_argument('--reassure_support_and_patch_combine',type=bool, default=False,
                        help='use REASSURE method to combine support and patch network')

        self.add_argument('--repair_radius',type=float, default=0.2, 
                          help='the radius of repairing datas or features')

        # training
        self.add_argument('--weight_decay', type=float, default=0.0, help='L2 regularization')
        self.add_argument('--k_coeff', type=float, default=1e-3, help='learning rate')
        self.add_argument('--accuracy_loss', type=str, choices=['L1', 'SmoothL1', 'MSE', 'CE'], default='CE',
                          help='canonical loss function for concrete points training')
        self.add_argument('--support_loss', type=str, choices=['CE','L2','SmoothL1'], default='L2',
                          help= 'canonical loss function for patch net training')
        self.add_argument('--sample_amount', type=int, default=5000,
                          help='specifically for data points sampling from spec')
        self.add_argument('--reset_params', type=literal_eval, default=False,
                          help='start with random weights or provided trained weights when available')
        self.add_argument('--train_datasize', type=int, default=5000, 
                          help='dataset size for training')
        self.add_argument('--test_datasize', type=int, default=2000,
                          help='dataset size for test')

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
            return ce(softmax(outs), labels)
            # return ce(outs, labels)

        def Bce_loss(outs: Tensor, labels: Tensor):
            bce = nn.BCEWithLogitsLoss()
            return bce(outs, labels)

        args.accuracy_loss = {
            'L1': nn.L1Loss(),
            'MSE': nn.MSELoss(),
            'CE': ce_loss
        }[args.accuracy_loss]

        args.support_loss = {
            'BCE' : Bce_loss,
            'L2': nn.MSELoss(),
            'L1': nn.L1Loss(),
            'CE': nn.CrossEntropyLoss(),
            'SmoothL1': nn.SmoothL1Loss(),


        }[args.support_loss]
        return
    pass

class MnistPoints(exp.ConcIns):
    """ Storing the concrete data points for one ACAS network sampled.
        Loads to CPU/GPU automatically.
    """
    @classmethod
    def load(cls, train: bool, device, trainnumber = None, testnumber = None,
            is_test_accuracy = False, 
            is_attack_testset_repaired = False, 
            is_attack_repaired = False):
        '''
        trainnumber: 训练集数据量
        testnumber: 测试集数据量
        is_test_accuracy: if True, 检测一般测试集的准确率
        is_attack_testset_repaired: if True, 检测一般被攻击测试集的准确率
        is_attack_repaired: if True, 检测被攻击数据的修复率
        三个参数只有一个为True
        '''
        suffix = 'train' if train else 'test'
        if train:
            fname = f'{suffix}_attack_data_full.pt'  # note that it is using original data
            # fname = f'{suffix}_norm00.pt'
            combine = torch.load(Path(MNIST_DATA_DIR, fname), device)
            inputs, labels = combine 
            inputs = inputs[:trainnumber]
            labels = labels[:trainnumber]
        else:
            if is_test_accuracy:
                fname = f'test_norm00.pt'
                combine = torch.load(Path(MNIST_DATA_DIR, fname), device)
                inputs, labels = combine
                inputs = inputs[:testnumber]
                labels = labels[:testnumber]
            
            elif is_attack_testset_repaired:
                fname = f'test_attack_data_full.pt'
                combine = torch.load(Path(MNIST_DATA_DIR, fname), device)
                inputs, labels = combine
                inputs = inputs[:testnumber]
                labels = labels[:testnumber]
            elif is_attack_repaired:
                fname = f'train_attack_data_full.pt'
                combine = torch.load(Path(MNIST_DATA_DIR, fname), device)
                inputs, labels = combine
                inputs = inputs[:testnumber]
                labels = labels[:testnumber]

            # clean_inputs, clean_labels = clean_combine
            # inputs = torch.cat((inputs[:testnumber], clean_inputs[:testnumber] ), dim=0)
            # labels = torch.cat((labels[:testnumber], clean_labels[:testnumber] ),  dim=0)
        
        assert len(inputs) == len(labels)
        return cls(inputs, labels)
    pass

def eval_test(net: MnistNet, testset: MnistPoints, categories=None) -> float:
    """ Evaluate accuracy on test set.
    :param categories: e.g., acas.AcasOut
    """
    with torch.no_grad():
        outs = net(testset.inputs)
        predicted = outs.argmax(dim=1)
        correct = (predicted == testset.labels).sum().item()
        # ratio = correct / len(testset)
        ratio = correct / len(testset.inputs)

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



def repair_mnist(args: Namespace, weight_clamp = False)-> Tuple[int, float, bool, float]:
    fname = 'mnist.pth'
    fpath = Path(MNIST_NET_DIR, fname)

    # train number是修复多少个点
    repairset = MnistPoints.load(train=True, device=device, trainnumber=args.repair_number)
    trainset = MnistPoints.load(train=True, device=device, trainnumber=args.train_datasize)
    testset = MnistPoints.load(train=False, device=device, testnumber=args.test_datasize,is_test_accuracy=True)
    attack_testset = MnistPoints.load(train=False, device=device, testnumber=args.test_datasize,is_attack_testset_repaired=True)

    net = MnistNet(dom=args.dom)
    net.to(device)
    net.load_state_dict(torch.load(fpath, map_location=device))

    # acc = eval_test(net, testset)
    acc = eval_test(net, repairset)

    # repair part
    # the number of repair patch network,which is equal to the number of properties
    # n_repair = MnistProp.LABEL_NUMBER
    n_repair = repairset.inputs.shape[0]
    input_shape = trainset.inputs.shape[1:]
    patch_lists = []
    
    for i in range(n_repair):
        patch_net = Mnist_patch_model(dom=args.dom,
            name = f'patch network {i}')
        patch_net.to(device)
        patch_lists.append(patch_net)
    logging.info(f'--Patch network: {patch_net}')


    # ABstraction part
    # set the box bound of the features, which the length of the box is 2*radius
    # bound_mins = feature - radius
    # bound_maxs = feature + radius

    # the steps are as follows:
    # 1.construct a mnist prop file include Mnistprop Module, which inherits from Oneprop
    # 2.use the features to construct the Mnistprop
    # 3.use the Mnistprop to construct the Andprop(join) 
    repairlist = [(data[0],data[1]) for data in zip(repairset.inputs, repairset.labels)]
    repair_prop_list = MnistProp.all_props(args.dom, DataList=repairlist, input_shape= input_shape,radius= args.repair_radius)

    # get the all props after join all l_0 ball feature property
    # TODO squeeze the property list, which is the same as the number of label
    all_props = AndProp(props=repair_prop_list)
    v = Bisecter(args.dom, all_props)

    in_lb, in_ub = all_props.lbub(device)
    in_bitmap = all_props.bitmap(device)

    def run_abs(net, batch_abs_lb: Tensor, batch_abs_ub: Tensor, batch_abs_bitmap: Tensor) -> Tensor:
        """ Return the safety distances over abstract domain. """
        batch_abs_ins = args.dom.Ele.by_intvl(batch_abs_lb, batch_abs_ub)
        batch_abs_outs = net(batch_abs_ins)
        return all_props.safe_dist(batch_abs_outs, batch_abs_bitmap)

        # judge the batch_inputs is in which region of property
    def get_bitmap(in_lb: Tensor, in_ub: Tensor, in_bitmap: Tensor, batch_inputs: Tensor):
        '''
        in_lb: n_prop * input
        in_ub: n_prop * input
        batch_inputs: batch * input
        '''
        batch_inputs_clone = batch_inputs.clone().unsqueeze_(1)
        batch_inputs_clone = batch_inputs_clone.expand(batch_inputs.shape[0], in_lb.shape[0], in_lb.shape[1])
        is_in = (batch_inputs_clone >= in_lb) & (batch_inputs_clone <= in_ub)
        is_in = is_in.all(dim=-1) # every input is in the region of property, batch * n_prop
        # convert to bitmap
        bitmap = torch.zeros((batch_inputs.shape[0], in_bitmap.shape[1]), device = device)
        # is in is a batch * in_bitmap.shape[0] tensor, in_bitmap.shape[1] is the number of properties
        # the every row of is_in is the bitmap of the input which row of in_bitmap is allowed
        for i in range(is_in.shape[0]):
            for j in range(is_in.shape[1]):
                if is_in[i][j]:
                    bitmap[i] = in_bitmap[j]
                    break
                else:
                    continue
        # how to use the vectoirzation to speed up the following code
        # tmp0 = in_bitmap.unsqueeze(0).expand(batch_inputs.shape[0], in_bitmap.shape[0], in_bitmap.shape[1]).to(device)
        # tmp0 = torch.zeros((batch_inputs.shape[0], in_bitmap.shape[0], in_bitmap.shape[1]), device = device)
        # tmp1 = is_in.unsqueeze(-1).expand(batch_inputs.shape[0], in_bitmap.shape[0], in_bitmap.shape[1]).to(device)
        # bitmap = torch.where(tmp1, tmp0, 1)
        # # tmp0[tmp1.nonzero(as_tuple=True)] = 1
        # # bitmap = tmp0.all(dim=-1)
        # a = bitmap.all(dim=-1)


        return bitmap

    # the number of repair patch network,which is equal to the number of properties 
    repair_net =  Netsum(args.dom, target_net = net, patch_nets= patch_lists, device=device)


    start = timer()

    if args.no_abs or args.no_refine:
        curr_abs_lb, curr_abs_ub, curr_abs_bitmap = in_lb, in_ub, in_bitmap
    else:
        # refine it at the very beginning to save some steps in later epochs
        # and use the refined bounds as the initial bounds for support network training
        
        curr_abs_lb, curr_abs_ub, curr_abs_bitmap = v.split(in_lb, in_ub, in_bitmap, net, args.refine_top_k,
                                                                tiny_width=args.tiny_width,
                                                                stop_on_k_all=args.start_abs_cnt, for_patch=False) #,for_support=False


    # train the patch and original network
    opti = Adam(repair_net.parameters(), lr=args.lr*0.1, weight_decay=args.weight_decay)
    # delate the parameters of the support network and the target network from the parameters of the repair network
    # TODO 这里的参数是不是有问题
    opti.param_groups[0]['params'] = opti.param_groups[0]['params'][12:]
    scheduler = args.scheduler_fn(opti)  # could be None

    # freeze the parameters of the original network for extracting features
    # for name, param in repair_net.named_parameters():
    #     if 'conv1' or 'conv2' or 'fc1' in name:
    #         param.requires_grad = False
    #         opti.param_groups[0]['params'].append(param)
    #         opti.param_groups[0]['lr'] = 0

    accuracies = []  # epoch 0: ratio
    repair_acc = []
    train_acc = []
    attack_test_acc = []
    certified = False
    epoch = 0

    # set the
    # feature_traindata.requires_grad = True
    # feature_trainset = MnistPoints(feature_traindata, trainset.labels)
    

    while True:
        # first, evaluate current model
        logging.info(f'[{utils.time_since(start)}] After epoch {epoch}:')
        if not args.no_pts:
            logging.info(f'Loaded {repairset.real_len()} points for repair.')
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
        # accuracies.append(eval_test(finally_net, testset))
        test_bitmap = get_bitmap(in_lb, in_ub, in_bitmap, testset.inputs)
        accuracies.append(eval_test(repair_net, testset, bitmap = test_bitmap))

        repairset_bitmap = get_bitmap(in_lb, in_ub, in_bitmap, repairset.inputs)
        repair_acc.append(eval_test(repair_net, repairset, bitmap = repairset_bitmap))

        trainset_bitmap = get_bitmap(in_lb, in_ub, in_bitmap, trainset.inputs)
        train_acc.append(eval_test(repair_net, trainset, bitmap = trainset_bitmap))

        attack_testset_bitmap = get_bitmap(in_lb, in_ub, in_bitmap, attack_testset.inputs)
        attack_test_acc.append(eval_test(repair_net, attack_testset, bitmap = attack_testset_bitmap))

        logging.info(f'Test set accuracy {accuracies[-1]}.')
        logging.info(f'repair set accuracy {accuracies[-1]}.')
        logging.info(f'train set accuracy {train_acc[-1]}.')
        logging.info(f'attacked test set accuracy {attack_test_acc[-1]}.')

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
            abs_loader = data.DataLoader(absset, batch_size=args.repair_batch_size, shuffle=True)
            nbatches = len(abs_loader)  # doesn't matter rewriting len(conc_loader), they are the same
            abs_loader = iter(abs_loader)

        total_loss = 0.
        # with torch.enable_grad():
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
    torch.save(repair_net.state_dict(), str(REPAIR_MODEL_DIR / f'trainset{args.datasize}-rapair_radius{args.repair_radius}-.pt'))
    logging.info(f'Accuracy at every epoch: {accuracies}')
    logging.info(f'After {epoch} epochs / {utils.pp_time(train_time)}, ' +
                 f'eventually the trained network got certified? {certified}, ' +
                 f'with {accuracies[-1]:.4f} accuracy on test set,' +
                 f'with {repair_acc[-1]:.4f} accuracy on repair set,' +
                 f'with {train_acc[-1]:.4f} accuracy on train set,' +
                 f'with {attack_test_acc[-1]:.4f} accuracy on attack test set.')
    # torch.save(repair_net.state_dict(), str(REPAIR_MODEL_DIR / '2_9.pt'))
    # net.save_nnet(f'./ART_{nid.x.numpy().tolist()}_{nid.y.numpy().tolist()}_repair2p_noclamp_epoch_{epoch}.nnet',
    #             mins = bound_mins, maxs = bound_maxs) 
    
    return epoch, train_time, certified, accuracies[-1]

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




def test_goal_repair(parser: MnistArgParser):
    """ Q1: Show that we can train previously unsafe networks to safe. """
    defaults = {
        # 'start_abs_cnt': 5000,
        # 'max_abs_cnt': 
        'batch_size': 128,  # to make it faster
        'min_epochs': 25,
        'max_epochs': 45
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

def test(lr:float = 0.005, repair_radius:float = 0.1, repair_number = 50,
         train_datasize = 5000, test_datasize = 2000, 
         accuracy_loss:str = 'CE'):
    test_defaults = {
        'exp_fn': 'test_goal_repair',
        'refine_top_k': 1,
        'repair_batch_size': 1,
        'start_abs_cnt': 500,
        'max_abs_cnt': 1000,
        'no_repair': False,
        'repair_number': repair_number,
        'train_datasize':train_datasize,
        'test_datasize':test_datasize,
        'repair_radius': repair_radius,
        'lr': lr,
        'accuracy_loss': accuracy_loss,
        'tiny_width': repair_radius*0.0001,

        
    }
    parser = MnistArgParser(RES_DIR, description='MNIST Correct by Construction')
    parser.set_defaults(**test_defaults)
    args, _ = parser.parse_known_args()

    exp_fn = globals()[args.exp_fn]
    start = timer()
    exp_fn(parser)

    logging.info(f'Total Cost Time: {timer() - start}s.\n\n\n')
    pass


if __name__ == '__main__':
    device = torch.device(f'cuda:0')
    # for lr in [0.005, 0.01]:
    #     for weight_decay in [0.0001, 0.00005]:
    #         # for k_coeff in [0.35, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
    #         for k_coeff in [0.4]:    
    #             for support_loss in ['SmoothL1', 'L2']:
    #                 for accuracy_loss in ['CE']:
    #                     # if lr == 0.005 and weight_decay == 0.0001 and k_coeff == 0.4 and support_loss == 'SmoothL1' and accuracy_loss == 'CE':
    #                     #     continue
    #                     # for repair_radius in [0.1, 0.05, 0.03, 0.01]:
    #                     test(lr=lr, weight_decay=weight_decay, k_coeff=k_coeff, repair_radius=0.1, support_loss=support_loss, accuracy_loss=accuracy_loss)
    test(lr=0.01, repair_radius=0.1, repair_number = 50,
         train_datasize = 5000, test_datasize = 2000, 
         accuracy_loss='CE')

    # # get the features of the dataset using mnist_net.split
    # model1, model2 = net.split()
    # with torch.no_grad():
    #     feature_traindata = model1(trainset.inputs)
    # feature_trainset = MnistPoints(feature_traindata, trainset.labels)
    # with torch.no_grad():
    #     feature_testdata = model1(testset.inputs)
    #     feature_attack_testdata = model1(attack_testset.inputs)
    # feature_testset = MnistPoints(feature_testdata, testset.labels)
    # feature_attack_testset = MnistPoints(feature_attack_testdata, attack_testset.labels)