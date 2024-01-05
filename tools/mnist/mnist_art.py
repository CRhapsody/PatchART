import logging
import math
from selectors import EpollSelector
import sys
from argparse import Namespace
from ast import literal_eval
from pathlib import Path
from typing import Optional, Tuple, List
from timeit import default_timer as timer
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils import data
import acas.acas_utils as acas


from art.prop import AndProp
from art.bisecter import Bisecter
from art import exp, utils
from mnist.mnist_utils import MnistNet_CNN_small, MnistNet_FNN_big, MnistNet_FNN_small, MnistProp, Mnist_patch_model

RES_DIR = Path(__file__).resolve().parent.parent / 'results' / 'mnist' / 'ART'
RES_DIR.mkdir(parents=True, exist_ok=True)
REPAIR_MODEL_DIR = Path(__file__).resolve().parent.parent / 'model' / 'mnist' / 'ART'
REPAIR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
MNIST_DATA_DIR = Path(__file__).resolve().parent.parent.parent / 'data' / 'MNIST' / 'processed'
MNIST_NET_DIR = Path(__file__).resolve().parent.parent.parent / 'model' /'mnist'
# MNIST_NET_DIR = Path(__file__).resolve().parent.parent / 'pgd' /'model' 



class MnistArgParser(exp.ExpArgParser):
    """ Parsing and storing all ACAS experiment configuration arguments. """

    def __init__(self, log_path: Optional[str], *args, **kwargs):
        super().__init__(log_path, *args, **kwargs)
        self.add_argument('--net', type=str, choices=['FNN_small', 'FNN_big', 'CNN_small'],
                          default='CNN_small', help='network architecture')
        self.add_argument('--repair_number', type=int, default=50,
                          help='the number of repair datas')    
        self.add_argument('--repair_batchsize', type=int, default=1,
                            help='the batchsize of repair datas')
        self.add_argument('--patch_size', type=str, default= 'big', 
                          choices=['big', 'small'], help='the size of patch network')
        self.add_argument('--repair_radius',type=float, default=0.2, 
                          help='the radius of repairing datas or features')
        self.add_argument('--train_datasize', type=int, default=10000, 
                          help='dataset size for training')
        self.add_argument('--test_datasize', type=int, default=2000,
                          help='dataset size for test')

        # training
        self.add_argument('--divided_repair', type=int, default=1, help='batch size for training')    

        self.add_argument('--accuracy_loss', type=str, choices=['L1', 'MSE', 'CE'], default='CE',
                          help='canonical loss function for concrete points training')
        self.add_argument('--sample_amount', type=int, default=5000,
                          help='specifically for data points sampling from spec')
        self.add_argument('--reset_params', type=literal_eval, default=False,
                          help='start with random weights or provided trained weights when available')

        # querying a verifier
        self.add_argument('--max_verifier_sec', type=int, default=200,
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


class MnistPoints(exp.ConcIns):
    """ Storing the concrete data points for one ACAS network sampled.
        Loads to CPU/GPU automatically.
    """
    @classmethod
    def load(cls, train: bool, device, 
            net,
            repairnumber = None,
            trainnumber = None, testnumber = None, radius = 0,
            is_test_accuracy = False, 
            is_attack_testset_repaired = False, 
            is_attack_repaired = False,
            is_origin_data = False):
        '''
        trainnumber: 训练集数据量
        testnumber: 测试集数据量
        radius: 修复数据的半径
        is_test_accuracy: if True, 检测一般测试集的准确率
        is_attack_testset_repaired: if True, 检测一般被攻击测试集的准确率
        is_attack_repaired: if True, 检测被攻击数据的修复率
        三个参数只有一个为True
        '''
        #_attack_data_full
        suffix = 'train' if train else 'test'
        if train:
            fname = f'train_norm00.pt'  # note that it is using original data
            # fname = f'{suffix}_norm00.pt'
            # mnist_train_norm00_dir = "/pub/data/chizm/"
            # combine = torch.load(mnist_train_norm00_dir+fname, device)
            combine = torch.load(Path(MNIST_DATA_DIR, fname), device)
            inputs, labels = combine 
            inputs = inputs[:trainnumber]
            labels = labels[:trainnumber]
        else:
            if is_test_accuracy:
                fname = f'test_norm00.pt'
                combine = torch.load(Path(MNIST_DATA_DIR, fname), device)
                inputs, labels = combine
                # inputs = inputs[:testnumber]
                # labels = labels[:testnumber]
            elif is_origin_data:
                fname = f'origin_data_{net}_{radius}.pt'
                combine = torch.load(Path(MNIST_DATA_DIR, fname), device)
                inputs, labels = combine
                inputs = inputs[:repairnumber]
                labels = labels[:repairnumber]
            
            elif is_attack_testset_repaired:
                fname = f'test_attack_data_full_{net}_{radius}.pt'
                combine = torch.load(Path(MNIST_DATA_DIR, fname), device)
                inputs, labels = combine
                inputs = inputs[:testnumber]
                labels = labels[:testnumber]
            elif is_attack_repaired:
                fname = f'train_attack_data_full_{net}_{radius}.pt'
                combine = torch.load(Path(MNIST_DATA_DIR, fname), device)
                inputs, labels = combine
                inputs = inputs[:repairnumber]
                labels = labels[:repairnumber]

            # clean_inputs, clean_labels = clean_combine
            # inputs = torch.cat((inputs[:testnumber], clean_inputs[:testnumber] ), dim=0)
            # labels = torch.cat((labels[:testnumber], clean_labels[:testnumber] ),  dim=0)
        
        assert len(inputs) == len(labels)
        return cls(inputs, labels)
    pass

def eval_test(net, testset: MnistPoints, bitmap: Tensor =None, categories=None) -> float:
    """ Evaluate accuracy on test set.
    :param categories: e.g., acas.AcasOut
    """
    with torch.no_grad():
        if bitmap is None:
            outs = net(testset.inputs)
        else:
            outs = net(testset.inputs, bitmap)
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


def train_mnist(args: Namespace) -> Tuple[int, float, bool, float]:
    """ The almost completed skeleton of training ACAS networks using ART.
    :return: trained_epochs, train_time, certified, final accuracies
    """
    fname = f'mnist_{args.net}.pth'
    fpath = Path(MNIST_NET_DIR, fname)

    # train number是修复多少个点
    # originalset = torch.load(Path(MNIST_DATA_DIR, f'origin_data_{args.repair_radius}.pt'), device)
    # originalset = MnistPoints(inputs=originalset[0], labels=originalset[1])
    originalset = MnistPoints.load(train=False, device=device, net=args.net, repairnumber=args.repair_number, radius=args.repair_radius,is_origin_data=True)
    repairset = MnistPoints.load(train=False, device=device, net=args.net, repairnumber=args.repair_number, radius=args.repair_radius,is_attack_repaired=True)
    trainset = MnistPoints.load(train=True, device=device, net=args.net, repairnumber=args.repair_number, trainnumber=args.train_datasize)
    testset = MnistPoints.load(train=False, device=device, net=args.net, repairnumber=args.repair_number, testnumber=args.test_datasize,is_test_accuracy=True)
    attack_testset = MnistPoints.load(train=False, device=device, net=args.net, repairnumber=args.repair_number, testnumber=args.test_datasize, radius=args.repair_radius,is_attack_testset_repaired=True)
    if args.net == 'CNN_small':
        net = MnistNet_CNN_small(dom=args.dom)
    elif args.net == 'FNN_big':
        net = MnistNet_FNN_big(dom=args.dom)
    elif args.net == 'FNN_small':
        net = MnistNet_FNN_small(dom=args.dom)
    net.to(device)
    net.load_state_dict(torch.load(fpath, map_location=device))

    # all_props = AndProp(nid.applicable_props(args.dom))
    # v = Bisecter(args.dom, all_props)
    n_repair = repairset.inputs.shape[0]
    input_shape = trainset.inputs.shape[1:]
    # repairlist = [(data[0],data[1]) for data in zip(repairset.inputs, repairset.labels)]
    # repair_prop_list = MnistProp.all_props(args.dom, DataList=repairlist, input_shape= input_shape,radius= args.repair_radius)
    repairlist = [(data[0],data[1]) for data in zip(originalset.inputs, originalset.labels)]
    repair_prop_list = MnistProp.all_props(args.dom, DataList=repairlist, input_shape= input_shape,radius= args.repair_radius)
    # get the all props after join all l_0 ball feature property
    all_props = AndProp(props=repair_prop_list)
    v = Bisecter(args.dom, all_props)


    def run_abs(batch_abs_lb: Tensor, batch_abs_ub: Tensor, batch_abs_bitmap: Tensor) -> Tensor:
        """ Return the safety distances over abstract domain. """
        batch_abs_ins = args.dom.Ele.by_intvl(batch_abs_lb, batch_abs_ub)
        batch_abs_outs = net(batch_abs_ins)
        return all_props.safe_dist(batch_abs_outs, batch_abs_bitmap)

    in_lb, in_ub = all_props.lbub(device)
    in_bitmap = all_props.bitmap(device)
    # in_lb = net.normalize_inputs(in_lb, bound_mins, bound_maxs)
    # in_ub = net.normalize_inputs(in_ub, bound_mins, bound_maxs)

    # already moved to GPU if necessary
    # trainset = AcasPoints.load(nid, train=True, device=device)
    # testset = AcasPoints.load(nid, train=False, device=device)
    torch.cuda.empty_cache()
    logging.info(f'--Test repair set accuracy {eval_test(net, repairset)}')
    start = timer()



    def get_curr_setmap(dataset, part):
        # recostruct the dataset and bitmap
        if part != args.divided_repair - 1:
            # curr_map = bitmap[int(part*bitmap.shape[0]/args.divided_repair):int((part+1)*bitmap.shape[0]/args.divided_repair)]
            curr_set = MnistPoints(dataset.inputs[int(part*len(dataset.inputs)/args.divided_repair):int((part+1)*len(dataset.inputs)/args.divided_repair)], dataset.labels[int(part*len(dataset.labels)/args.divided_repair):int((part+1)*len(dataset.labels)/args.divided_repair)])
        else:
            # curr_map = bitmap[int(part*bitmap.shape[0]/args.divided_repair):]
            curr_set = MnistPoints(dataset.inputs[int(part*len(dataset.inputs)/args.divided_repair):], dataset.labels[int(part*len(dataset.inputs)/args.divided_repair):])
        # return curr_set, curr_map
        return curr_set



    for o in range(args.divided_repair):
        accuracies = []  # epoch 0: ratio
        repair_acc = []
        train_acc = []
        attack_test_acc = []
        certified = False
        epoch = 0
        torch.cuda.empty_cache()
        divide_repair_number = int(n_repair/args.divided_repair)

        if o != args.divided_repair - 1:
            curr_abs_lb, curr_abs_ub, curr_abs_bitmap = in_lb[o*divide_repair_number:(o+1)*divide_repair_number], in_ub[o*divide_repair_number:(o+1)*divide_repair_number], in_bitmap[o*divide_repair_number:(o+1)*divide_repair_number]
            # if args.net == 'FNN_big' or args.net == 'FNN_small':
            #     opti.param_groups[0]['params'] = param_copy[14+6*o*divide_repair_number:14+6*(o+1)*divide_repair_number]
        else:
            curr_abs_lb, curr_abs_ub, curr_abs_bitmap = in_lb[o*divide_repair_number:], in_ub[o*divide_repair_number:], in_bitmap[o*divide_repair_number:]


    if args.no_abs or args.no_refine:
        curr_abs_lb, curr_abs_ub, curr_abs_bitmap = in_lb, in_ub, in_bitmap
    else:
        # refine it at the very beginning to save some steps in later epochs
        curr_abs_lb, curr_abs_ub, curr_abs_bitmap = v.split(in_lb, in_ub, in_bitmap, net, args.refine_top_k,
                                                            # tiny_width=args.tiny_width,
                                                            stop_on_k_all=args.start_abs_cnt)
    curr_repairset = get_curr_setmap(repairset, o)
    curr_attack_testset = get_curr_setmap(attack_testset, o)


    opti = Adam(net.parameters(), lr=args.lr)
    scheduler = args.scheduler_fn(opti)  # could be None

    while True:
        train_time = timer() - start
        if train_time > 3600:
            logging.warning(f'Training time {train_time} exceeds 1 hour, timeout.')
            logging.info(f'save model')
            torch.save(net.state_dict(), str(REPAIR_MODEL_DIR / f'Mnist-{args.net}-repair_number{args.repair_number}-rapair_radius{args.repair_radius}.pt'))
            return epoch, train_time, certified, accuracies[-1]
        # first, evaluate current model
        logging.info(f'[{utils.time_since(start)}] After epoch {epoch}:')
        if not args.no_pts:
            logging.info(f'Loaded {trainset.real_len()} points for training.')
        if not args.no_abs:
            logging.info(f'Loaded {len(curr_abs_lb)} abstractions for training.')
            with torch.no_grad():
                full_dists = run_abs(curr_abs_lb, curr_abs_ub, curr_abs_bitmap)
            logging.info(f'min loss {full_dists.min()}, max loss {full_dists.max()}.')
            if full_dists.max() <= 0.:
                certified = True
                logging.info(f'All {len(curr_abs_lb)} abstractions certified.')
            else:
                _, worst_idx = full_dists.max(dim=0)
                logging.debug(f'Max loss at LB: {curr_abs_lb[worst_idx]}, UB: {curr_abs_ub[worst_idx]}, rule: {curr_abs_bitmap[worst_idx]}.')

        accuracies.append(eval_test(net, testset))
        logging.info(f'Test set accuracy {accuracies[-1]}.')



        accuracies.append(eval_test(net, testset))
        
        repair_acc.append(eval_test(net, curr_repairset))

        train_acc.append(eval_test(net, trainset))

        attack_test_acc.append(eval_test(net, curr_attack_testset))

        logging.info(f'Test set accuracy {accuracies[-1]}.')
        logging.info(f'repair set accuracy {repair_acc[-1]}.')
        logging.info(f'train set accuracy {train_acc[-1]}.')
        logging.info(f'attacked test set accuracy {attack_test_acc[-1]}.')

        # check termination
        if len(repair_acc) >= 2:
            if (repair_acc[-1] == 1.0 and attack_test_acc[-1] == 1.0) or certified:
            # all safe and sufficiently trained
                break
        elif certified or (repair_acc[-1] == 1.0 and attack_test_acc[-1] == 1.0):
            break

        if epoch >= args.max_epochs:
            break

        epoch += 1
        certified = False
        logging.info(f'\n[{utils.time_since(start)}] Starting epoch {epoch}:')

        absset = exp.AbsIns(curr_abs_lb, curr_abs_ub, curr_abs_bitmap)
        # if args.repair_number < 100:
        batch_size = 100
        # else:
        #     batch_size = 100
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
            conc_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
            nbatches = len(conc_loader)
            conc_loader = iter(conc_loader)
        if not args.no_abs:
            abs_loader = data.DataLoader(absset, batch_size=batch_size, shuffle=True)
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
                safe_loss = batch_dists.mean()  # L1, need to upgrade to batch_worsts to unlock loss other than L1
                total_loss += safe_loss.item()
                batch_loss += safe_loss
            logging.debug(f'Epoch {epoch}: {i / nbatches * 100 :.2f}%. Batch loss {batch_loss.item()}')
            batch_loss.backward()
            opti.step()

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
    # timeout
    torch.save(net.state_dict(), str(REPAIR_MODEL_DIR / f'Mnist-{args.net}-repair_number{args.repair_number}-rapair_radius{args.repair_radius}.pt'))
    logging.info(f'Accuracy at every epoch: {accuracies}')
    logging.info(f'After {epoch} epochs / {utils.pp_time(train_time)}, ' +
                 f'eventually the trained network got certified? {certified}, ' +
                 f'with {accuracies[-1]:.4f} accuracy on test set.')
    return epoch, train_time, certified, accuracies[-1]


def _run(args: Namespace):
    """ Run for different networks with specific configuration. """
    res = []

    logging.info(f'For {args.net}')
    outs = train_mnist(args)
    res.append(outs)

    avg_res = torch.tensor(res).mean(dim=0)
    avg_res = torch.tensor(res).mean(dim=0)
    logging.info(f'=== Avg <epochs, train_time, certified, accuracy> for {args.net} networks:')
    logging.info(avg_res)
    return


def test_goal_safety(parser: MnistArgParser):
    """ Q1: Show that we can train previously unsafe networks to safe. """
    defaults = {
        # 'start_abs_cnt': 5000,
        'batch_size': 50,  # to make it faster
        'min_epochs': 15,
        'max_epochs': 25,
    }
    parser.set_defaults(**defaults)
    args = parser.parse_args()

    logging.info(utils.fmt_args(args))
    # nids = acas.AcasNetID.goal_safety_ids(args.dom)
    _run(args)
    return


def test_goal_accuracy(parser: MnistArgParser):
    """ Q2: Show that the safe-by-construction overhead on accuracy is mild. """
    defaults = {
        # 'start_abs_cnt': 5000,
        'batch_size': 100,  # to make it faster
        'min_epochs': 25,
        'max_epochs': 35,

    }
    parser.set_defaults(**defaults)
    args = parser.parse_args()

    logging.info(utils.fmt_args(args))
    nids = acas.AcasNetID.goal_accuracy_ids(args.dom)
    _run(nids, args)
    return

def test(lr:float = 0.005, net:str = 'CNN_small',repair_radius:float = 0.1, repair_number = 200, refine_top_k = 50,
         train_datasize = 200, test_datasize = 2000, 
         accuracy_loss:str = 'CE'):
    test_defaults = {
        'exp_fn': 'test_goal_safety',
        'net': net,
        'no_refine': False,
        
        'refine_top_k': refine_top_k,
        'repair_batch_size': repair_number,
        'repair_number': repair_number,
        'train_datasize':train_datasize,
        'test_datasize': test_datasize,
        'repair_radius': repair_radius,
        'lr': lr,
        'accuracy_loss': accuracy_loss,
        'tiny_width': repair_radius*0.0001,
        'divided_repair': math.ceil(repair_number/50),
        # 'start_abs_cnt': 100,
        # 'max_abs_cnt': 2000,
        # 'no_refine': True
    }
    if net == 'FNN_small':
        test_defaults['divided_repair']= math.ceil(repair_number/100)
        test_defaults['start_abs_cnt']= 500
        test_defaults['max_abs_cnt']= 1800
    elif net == 'FNN_big':
        test_defaults['divided_repair']= math.ceil(repair_number/100)
        test_defaults['start_abs_cnt']= 500
        test_defaults['max_abs_cnt']= 1200
    elif net == 'CNN_small':
        test_defaults['divided_repair']= math.ceil(repair_number/5)
        test_defaults['start_abs_cnt']= 20
        test_defaults['max_abs_cnt']= 50
    parser = MnistArgParser(RES_DIR, description='Mnist Correct by Construction')
    parser.set_defaults(**test_defaults)
    args, _ = parser.parse_known_args()

    exp_fn = globals()[args.exp_fn]
    start = timer()
    exp_fn(parser)

    logging.info(f'Total Cost Time: {timer() - start}s.\n\n\n')
    pass

if __name__ == '__main__':
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

    # for net in ['FNN_small', 'FNN_big', 'CNN_small']:
    for net in ['FNN_small']:
        # for patch_size in ['small', 'big']:
        # for patch_size in ['big']:
            # for radius in [0.3]:
        for radius in [0.1]: #,0.1,0.3
                # for repair_number,test_number in zip([100],[1000]):
                # for repair_number,test_number in zip([1000],[10000]):

            # for repair_number,test_number in zip([50,100,200],[500,1000,2000]):
            for repair_number,test_number in zip([50],[500]):
            # for repair_number,test_number in zip([50,100,200,500],[500,1000,2000,5000]):
            # for repair_number,test_number in zip([500,1000],[5000,10000]):
                # if radius == 0.1 and (repair_number == 50 ):
                #     continue
                test(lr=0.1, net=net, repair_radius=radius, repair_number = repair_number, refine_top_k= 50, 
         train_datasize = 10000, test_datasize = test_number, 
         accuracy_loss='CE')