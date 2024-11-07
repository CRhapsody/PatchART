import sys
from pathlib import Path
from typing import List, Optional, Tuple, Iterable, Sequence, Union
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from torch import Tensor, nn
from DiffAbs.DiffAbs import AbsDom, AbsEle, deeppoly
from cifar10.cifar10_utils import Resnet_model, Vgg_model
from mnist.mnist_utils import MnistNet_FNN_small, MnistNet_FNN_big, MnistNet_CNN_small
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tinyimagenet.tinyimagenet_utils import Wide_Resnet_101_2_model, Resnet_152_model
from pgd.PGDAttack import PGD

"""
This module implements Randomized Smoothing applied to classifier predictions.

| Paper link: https://arxiv.org/abs/1902.02918
"""
# from __future__ import absolute_import, division, print_function, unicode_literals

from abc import ABC
import logging
from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm
from tqdm.auto import tqdm

ART_NUMPY_DTYPE = np.float32

logger = logging.getLogger(__name__)


class RandomizedSmoothingMixin(ABC):
    """
    Implementation of Randomized Smoothing applied to classifier predictions and gradients, as introduced
    in Cohen et al. (2019).

    | Paper link: https://arxiv.org/abs/1902.02918
    """

    def __init__(
        self,
        sample_size: int,
        *args,
        scale: float = 0.1,
        alpha: float = 0.001,
        **kwargs,
    ) -> None:
        """
        Create a randomized smoothing wrapper.

        :param sample_size: Number of samples for smoothing.
        :param scale: Standard deviation of Gaussian noise added.
        :param alpha: The failure probability of smoothing.
        """
        super().__init__(*args, **kwargs)  # type: ignore
        self.sample_size = sample_size
        self.scale = scale
        self.alpha = alpha

    def _predict_classifier(self, x: np.ndarray, batch_size: int, training_mode: bool, **kwargs) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.

        :param x: Input samples.
        :param batch_size: Size of batches.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        raise NotImplementedError

    def predict(self, x: np.ndarray, batch_size: int = 128, verbose: bool = False, **kwargs) -> np.ndarray:
        """
        Perform prediction of the given classifier for a batch of inputs, taking an expectation over transformations.

        :param x: Input samples.
        :param batch_size: Batch size.
        :param verbose: Display training progress bar.
        :param is_abstain: True if function will abstain from prediction and return 0s. Default: True
        :type is_abstain: `boolean`
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        from scipy.stats import binom_test

        is_abstain = kwargs.get("is_abstain")
        if is_abstain is not None and not isinstance(is_abstain, bool):  # pragma: no cover
            raise ValueError("The argument is_abstain needs to be of type bool.")
        if is_abstain is None:
            is_abstain = True

        logger.info("Applying randomized smoothing.")
        n_abstained = 0
        prediction = []
        for x_i in tqdm(x, desc="Randomized smoothing", disable=not verbose):
            # get class counts
            counts_pred = self._prediction_counts(x_i, batch_size=batch_size)
            top = counts_pred.argsort()[::-1]
            count1 = np.max(counts_pred)
            count2 = counts_pred[top[1]]

            # predict or abstain
            smooth_prediction = np.zeros(counts_pred.shape)
            if (not is_abstain) or (binom_test(count1, count1 + count2, p=0.5) <= self.alpha):
                smooth_prediction[np.argmax(counts_pred)] = 1
            elif is_abstain:
                n_abstained += 1

            prediction.append(smooth_prediction)
        if n_abstained > 0:
            logger.info("%s prediction(s) abstained.", n_abstained)
        return np.array(prediction)

    def _fit_classifier(self, x: np.ndarray, y: np.ndarray, batch_size: int, nb_epochs: int, **kwargs) -> None:
        """
         Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :param batch_size: Batch size.
        :param nb_epochs: Number of epochs to use for training.
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
               and providing it takes no effect.
        """
        raise NotImplementedError

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int = 128, nb_epochs: int = 10, **kwargs) -> None:
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :param batch_size: Batch size.
        :param nb_epochs: Number of epochs to use for training.
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
               and providing it takes no effect.
        """
        self._fit_classifier(x, y, batch_size=batch_size, nb_epochs=nb_epochs, **kwargs)

    def certify(self, x: np.ndarray, n: int, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes certifiable radius around input `x` and returns radius `r` and prediction.

        :param x: Sample input with shape as expected by the model.
        :param n: Number of samples for estimate certifiable radius.
        :param batch_size: Batch size.
        :return: Tuple of length 2 of the selected class and certified radius.
        """
        prediction = []
        radius = []

        for x_i in x:

            # get sample prediction for classification
            counts_pred = self._prediction_counts(x_i, n=self.sample_size, batch_size=batch_size)
            class_select = int(np.argmax(counts_pred))

            # get sample prediction for certification
            counts_est = self._prediction_counts(x_i, n=n, batch_size=batch_size)
            count_class = counts_est[class_select]

            prob_class = self._lower_confidence_bound(count_class, n)

            if prob_class < 0.5:
                prediction.append(-1)
                radius.append(0.0)
            else:
                prediction.append(class_select)
                radius.append(self.scale * norm.ppf(prob_class))

        return np.array(prediction), np.array(radius)

    def _noisy_samples(self, x: np.ndarray, n: Optional[int] = None) -> np.ndarray:
        """
        Adds Gaussian noise to `x` to generate samples. Optionally augments `y` similarly.

        :param x: Sample input with shape as expected by the model.
        :param n: Number of noisy samples to create.
        :return: Array of samples of the same shape as `x`.
        """
        # set default value to sample_size
        if n is None:
            n = self.sample_size

        # augment x
        x = np.expand_dims(x, axis=0)
        x = np.repeat(x, n, axis=0)
        x = x + np.random.normal(scale=self.scale, size=x.shape).astype(ART_NUMPY_DTYPE)

        return x

    def _prediction_counts(self, x: np.ndarray, n: Optional[int] = None, batch_size: int = 128) -> np.ndarray:
        """
        Makes predictions and then converts probability distribution to counts.

        :param x: Sample input with shape as expected by the model.
        :param n: Number of noisy samples to create.
        :param batch_size: Size of batches.
        :return: Array of counts with length equal to number of columns of `x`.
        """
        # sample and predict
        x_new = self._noisy_samples(x, n=n)
        predictions = self._predict_classifier(x=x_new, batch_size=batch_size, training_mode=False)

        # convert to binary predictions
        idx = np.argmax(predictions, axis=-1)
        pred = np.zeros(predictions.shape)
        pred[np.arange(pred.shape[0]), idx] = 1

        # get class counts
        counts = np.sum(pred, axis=0)

        return counts

    def _lower_confidence_bound(self, n_class_samples: int, n_total_samples: int) -> float:
        """
        Uses Clopper-Pearson method to return a (1-alpha) lower confidence bound on bernoulli proportion

        :param n_class_samples: Number of samples of a specific class.
        :param n_total_samples: Number of samples for certification.
        :return: Lower bound on the binomial proportion w.p. (1-alpha) over samples.
        """
        from statsmodels.stats.proportion import proportion_confint

        return proportion_confint(n_class_samples, n_total_samples, alpha=2 * self.alpha, method="beta")[0]
    

def test_set_sample(model_t, radius,dataset,
                    sample_n = 100, std = 0.1, device = 'cuda:0'):
    '''
    Mnist: 0.05,10 即可
    resnet: 4 , std: 0.2, sample_n: 10; 8, std: 0.35, sample_n: 10 (60%); testset, 0.05-0.1 (0.2:5%)
    vgg: 4, std: 0.2, sample_n: 10; 8, std: 0.3, sample_n: 10 (66%); testset, 0.05-0.1 (0.2:5%)
    '''
    # test_set = torch.load('/home/chizm/AdvRepair/data/cifar10/test_norm.pt')
    # test_set = torch.load('/home/chizm/AdvRepair/data/cifar10/train_attack_data_full_resnet18_4.pt')
    # test_set = torch.load('/home/chizm/AdvRepair/data/cifar10/train_attack_data_full_resnet18_8.pt')

    # test_set = torch.load('/home/chizm/AdvRepair/data/cifar10/train_attack_data_full_vgg19_4.pt')
    # test_set = torch.load('/home/chizm/AdvRepair/data/cifar10/train_attack_data_full_vgg19_8.pt')

    if dataset == 'mnist':
        repair_data, repair_label = torch.load(f'/home/chizm/AdvRepair/data/MNIST/processed/train_attack_data_full_{model_t}_{radius}.pt')
        general_repair_data, general_repair_label = torch.load(f'/home/chizm/AdvRepair/data/MNIST/processed/test_attack_data_full_{model_t}_{radius}.pt')
        test_data,test_label = torch.load('/home/chizm/AdvRepair/data/MNIST/processed/test_norm00.pt')

    if model_t == 'FNN_small':
        model = MnistNet_FNN_small(dom=deeppoly)
        model_state_dict = torch.load('model/mnist/mnist_FNN_small.pth', map_location=device)
        model.load_state_dict(model_state_dict)
    elif model_t == 'FNN_big':
        model = MnistNet_FNN_big(dom=deeppoly)
        model_state_dict = torch.load('model/mnist/mnist_FNN_big.pth', map_location=device)
        model.load_state_dict(model_state_dict)
    elif model_t == 'CNN_small':
        model = MnistNet_CNN_small(dom=deeppoly)
        model_state_dict = torch.load('model/mnist/mnist_CNN_small.pth', map_location=device)
        model.load_state_dict(model_state_dict)
    model.to(device)
    # model = Resnet_model(dom=deeppoly).to(device)
    # model_state_dict = torch.load('model/cifar10/resnet18.pth', map_location=device)
    # model = Vgg_model(dom=deeppoly).to(device)
    # model_state_dict = torch.load('model/cifar10/vgg19.pth', map_location=device)
    # model.load_state_dict(model_state_dict)
    
    # make dataset
    testset =  torch.utils.data.TensorDataset(test_data, test_label)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers= 0)
    repairset =  torch.utils.data.TensorDataset(repair_data, repair_label)
    repairloader = torch.utils.data.DataLoader(repairset, batch_size=64, shuffle=False, num_workers= 0)
    general_repairset =  torch.utils.data.TensorDataset(general_repair_data, general_repair_label)
    general_repairloader = torch.utils.data.DataLoader(general_repairset, batch_size=64, shuffle=False, num_workers= 0)
    
    lower = test_data.min().to(device)
    upper = test_data.max().to(device)
    
    def sample(sample_n,std,lower= lower,upper = upper,
               dataloader = testloader, model = model, loader = 'testloader'):
    # sample the data
        correct_sum = 0


        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs_sample = torch.repeat_interleave(inputs, sample_n, dim=0)
            labels_sample = torch.repeat_interleave(labels, sample_n, dim=0)
            inputs_sample = inputs_sample + torch.normal(mean=0.0, std=std, size=inputs_sample.shape).to(device)
            inputs_sample = inputs_sample.clamp(lower, upper)

            with torch.no_grad():
                outputs = model(inputs_sample)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels_sample).sum().item()
            correct_sum += correct
        # construct the text file
        
        with open(f'/home/chizm/AdvRepair/results/sa.txt','a') as f:
            f.write(f'model:{model_t}, radius:{radius}, dataloader: {loader} sample_n:{sample_n}, std:{std}, correct_rate:{correct_sum/sample_n}\n')    
        print(f'model:{model_t}, radius:{radius}, dataloader: {loader} sample_n:{sample_n}, std:{std}, correct_rate:',correct_sum/(sample_n*len(dataloader)*64))

    sample(sample_n=sample_n,std=std,loader = 'testloader')
    sample(sample_n=sample_n,std=std,loader = 'repairloader',dataloader = repairloader)
    sample(sample_n=sample_n,std=std,loader = 'general_repairloader',dataloader = general_repairloader)



def get_model(model_t,device):
    if model_t == 'FNN_small':
        model = MnistNet_FNN_small(dom=deeppoly)
        model_state_dict = torch.load('model/mnist/mnist_FNN_small.pth', map_location=device)
        model.load_state_dict(model_state_dict)
    elif model_t == 'FNN_big':
        model = MnistNet_FNN_big(dom=deeppoly)
        model_state_dict = torch.load('model/mnist/mnist_FNN_big.pth', map_location=device)
        model.load_state_dict(model_state_dict)
    elif model_t == 'CNN_small':
        model = MnistNet_CNN_small(dom=deeppoly)
        model_state_dict = torch.load('model/mnist/mnist_CNN_small.pth', map_location=device)
        model.load_state_dict(model_state_dict)
    elif model_t == 'resnet18':
        model = Resnet_model(dom=deeppoly).to(device)
        model_state_dict = torch.load('model/cifar10/resnet18.pth', map_location=device)
        model.load_state_dict(model_state_dict)
    elif model_t == 'vgg19':
        model = Vgg_model(dom=deeppoly).to(device)
        model_state_dict = torch.load('model/cifar10/vgg19.pth', map_location=device)
        model.load_state_dict(model_state_dict)
    elif model_t == 'wide_resnet101_2':
        model = Wide_Resnet_101_2_model(dom=deeppoly).to(device)
        model_state_dict = torch.load('model/tiny_imagenet/wide_resnet101_2.pth', map_location=device)
        model.load_state_dict(model_state_dict)
    elif model_t == 'resnet152':
        model = Resnet_152_model(dom=deeppoly).to(device)
        model_state_dict = torch.load('model/tiny_imagenet/resnet152.pth', map_location=device)
        model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    return model

def get_test_dataset(dataset,device):
    if dataset == 'mnist':
        test_data,test_label = torch.load('/home/chizm/AdvRepair/data/MNIST/processed/test_norm00.pt')
    elif dataset == 'cifar10':
        test_data,test_label = torch.load('/home/chizm/AdvRepair/data/cifar10/test_norm.pt')
    elif dataset == 'tinyimagenet':
        test_data,test_label = torch.load('/home/chizm/PatchART/data/tiny_imagenet/test.pt')
    testset =  torch.utils.data.TensorDataset(test_data, test_label)
    return testset

def get_trades_model(model_t,radius,device = 'cuda:0'):
    if model_t == 'resnet18':
        model = Resnet_model().to(device)
        model_state_dict = torch.load(f'/home/chizm/PatchART/tools/cifar/trade-cifar/best_resnet18_{radius}_1000.pt', map_location=device)
        model.load_state_dict(model_state_dict)
    elif model_t == 'vgg19':
        model = Vgg_model().to(device)
        model_state_dict = torch.load(f'/home/chizm/PatchART/tools/cifar/trade-cifar/best_vgg19_{radius}_1000.pt', map_location=device)
        model.load_state_dict(model_state_dict)
    elif model_t == 'wide_resnet101_2':
        model = Wide_Resnet_101_2_model().to(device)
        model_state_dict = torch.load(f'/home/chizm/PatchART/trades_train/resnet152_{radius}_1000.pth', map_location=device)
        model.load_state_dict(model_state_dict)
    elif model_t == 'resnet152':
        model = Resnet_152_model().to(device)
        model_state_dict = torch.load(f'/home/chizm/PatchART/trades_train/resnet152_{radius}_1000.pth', map_location=device)
        model.load_state_dict(model_state_dict)

def generate_adv_sample(model_t, radius,dataset, device = 'cuda:1'):
    testset = get_test_dataset(dataset,device)
    model = get_model(model_t,device)
    trades_model = get_trades_model(model_t,radius,device)
    pgd = PGD(model,eps=radius/255,alpha=radius/(4*255),steps=100,random_start=True)
    from torchattacks import AutoAttack
    attack = AutoAttack(model, norm='Linf', eps=radius/255, version='standard', n_classes=10, seed=None, verbose=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers= 0)
    adv_storage = []
    labels_storage = []
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # adv = pgd.forward(inputs,labels)
        adv = attack(inputs,labels)
        pred = model(adv)
        _, predicted = torch.max(pred, 1)
        bit = (predicted == labels)

        # with torch.no_grad():
        #     trade_pred = trades_model(adv)
        # _, trade_predicted = torch.max(trade_pred, 1)
        # trade_bit = (trade_predicted == labels)

        # # get the intersection of the two bit
        # both_wrong_bit = ~bit & ~trade_bit
        # both_wrong_bit_rate = both_wrong_bit.sum().item()/64
        # print (f'batch {i} both wrong rate:',both_wrong_bit_rate)

        # if both 

        adv_storage.append(adv[~bit])
        # while predicted == labels:
        #     adv = pgd.forward(adv,labels)
        #     pred = model(adv)
        #     _, predicted = torch.max(pred, 1)
        # assert predicted != labels
        # adv_storage.append(adv)
        labels_storage.append(labels[~bit])
    adv_storage = torch.cat(adv_storage)
    labels_storage = torch.cat(labels_storage)
    if dataset == 'mnist':
        torch.save((adv_storage,labels_storage),f'/home/chizm/PatchART/data/{dataset}/test_adv_{model_t}_{radius}.pt')
    elif dataset == 'cifar10':
        torch.save((adv_storage,labels_storage),f'/home/chizm/PatchART/data/{dataset}/test_adv_{model_t}_{radius}.pt')
    elif dataset == 'tinyimagenet':
        torch.save((adv_storage,labels_storage),f'/home/chizm/PatchART/data/tiny_imagenet/test_adv_{model_t}_{radius}.pt')     
    print('save adv samples')



def set_sample(model_t, radius,dataset,
                    sample_n = 100, std = 0.1, device = 'cuda:1'):
    '''
    Mnist: 0.05,10 即可
    resnet: 4 , std: 0.2, sample_n: 10; 8, std: 0.35, sample_n: 10 (60%); testset, 0.05-0.1 (0.2:5%)
    vgg: 4, std: 0.2, sample_n: 10; 8, std: 0.3, sample_n: 10 (66%); testset, 0.05-0.1 (0.2:5%)
    '''
    # test_set = torch.load('/home/chizm/AdvRepair/data/cifar10/test_norm.pt')
    # test_set = torch.load('/home/chizm/AdvRepair/data/cifar10/train_attack_data_full_resnet18_4.pt')
    # test_set = torch.load('/home/chizm/AdvRepair/data/cifar10/train_attack_data_full_resnet18_8.pt')

    # test_set = torch.load('/home/chizm/AdvRepair/data/cifar10/train_attack_data_full_vgg19_4.pt')
    # test_set = torch.load('/home/chizm/AdvRepair/data/cifar10/train_attack_data_full_vgg19_8.pt')

    if dataset == 'mnist':
        repair_data, repair_label = torch.load(f'/home/chizm/AdvRepair/data/MNIST/processed/train_attack_data_full_{model_t}_{radius}.pt')
        general_repair_data, general_repair_label = torch.load(f'/home/chizm/AdvRepair/data/MNIST/processed/test_attack_data_full_{model_t}_{radius}.pt')
        test_data,test_label = torch.load('/home/chizm/AdvRepair/data/MNIST/processed/test_norm00.pt')
    elif dataset == 'cifar10':
        repair_data, repair_label = torch.load(f'/home/chizm/AdvRepair/data/cifar10/train_attack_data_full_{model_t}_{radius}.pt')
        general_repair_data, general_repair_label = torch.load(f'/home/chizm/AdvRepair/data/cifar10/test_attack_data_full_{model_t}_{radius}.pt')
        test_data,test_label = torch.load('/home/chizm/PatchART/data/cifar10/test_norm.pt')
        adv_test_data,adv_test_label = torch.load(f'/home/chizm/PatchART/data/cifar10/test_adv_{model_t}_{radius}.pt')
    elif dataset == 'tinyimagenet':
        repair_data, repair_label = torch.load(f'/home/chizm/PatchART/data/tiny_imagenet/train_attack_data_full_{model_t}_{radius}.pt')
        general_repair_data, general_repair_label = torch.load(f'/home/chizm/PatchART/data/tiny_imagenet/test_attack_data_full_{model_t}_{radius}.pt')
        test_data,test_label = torch.load('/home/chizm/PatchART/data/tiny_imagenet/test.pt')
        adv_test_data,adv_test_label = torch.load(f'/home/chizm/PatchART/data/tiny_imagenet/test_adv_{model_t}_{radius}.pt')
    
    if model_t == 'FNN_small':
        model = MnistNet_FNN_small(dom=deeppoly)
        model_state_dict = torch.load('model/mnist/mnist_FNN_small.pth', map_location=device)
        model.load_state_dict(model_state_dict)
    elif model_t == 'FNN_big':
        model = MnistNet_FNN_big(dom=deeppoly)
        model_state_dict = torch.load('model/mnist/mnist_FNN_big.pth', map_location=device)
        model.load_state_dict(model_state_dict)
    elif model_t == 'CNN_small':
        model = MnistNet_CNN_small(dom=deeppoly)
        model_state_dict = torch.load('model/mnist/mnist_CNN_small.pth', map_location=device)
        model.load_state_dict(model_state_dict)
    elif model_t == 'resnet18':
        model = Resnet_model(dom=deeppoly).to(device)
        model_state_dict = torch.load('model/cifar10/resnet18.pth', map_location=device)
        model.load_state_dict(model_state_dict)
    elif model_t == 'vgg19':
        model = Vgg_model(dom=deeppoly).to(device)
        model_state_dict = torch.load('model/cifar10/vgg19.pth', map_location=device)
        model.load_state_dict(model_state_dict)
    elif model_t == 'wide_resnet101_2':
        model = Wide_Resnet_101_2_model(dom=deeppoly).to(device)
        model_state_dict = torch.load('model/tiny_imagenet/wide_resnet101_2.pth', map_location=device)
        model.load_state_dict(model_state_dict)
    elif model_t == 'resnet152':
        model = Resnet_152_model(dom=deeppoly).to(device)
        model_state_dict = torch.load('model/tiny_imagenet/resnet152.pth', map_location=device)
        model.load_state_dict(model_state_dict)

    model.to(device)
    model.eval()
    # model = Resnet_model(dom=deeppoly).to(device)
    # model_state_dict = torch.load('model/cifar10/resnet18.pth', map_location=device)
    # model = Vgg_model(dom=deeppoly).to(device)
    # model_state_dict = torch.load('model/cifar10/vgg19.pth', map_location=device)
    # model.load_state_dict(model_state_dict)
    
    # make dataset
    testset =  torch.utils.data.TensorDataset(test_data, test_label)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers= 0)
    repairset =  torch.utils.data.TensorDataset(repair_data, repair_label)
    repairloader = torch.utils.data.DataLoader(repairset, batch_size=64, shuffle=False, num_workers= 0)
    general_repairset =  torch.utils.data.TensorDataset(general_repair_data, general_repair_label)
    general_repairloader = torch.utils.data.DataLoader(general_repairset, batch_size=64, shuffle=False, num_workers= 0)
    
    if dataset == 'tinyimagenet' or dataset == 'cifar10':
        adv_testset =  torch.utils.data.TensorDataset(adv_test_data, adv_test_label)
        adv_testloader = torch.utils.data.DataLoader(adv_testset, batch_size=64, shuffle=False, num_workers= 0)
    lower = test_data.min().to(device)
    upper = test_data.max().to(device)
    
    def sample(sample_n,std,lower= lower,upper = upper,
               dataloader = testloader, model = model, loader = 'testloader'):
    # sample the data
        correct_sum = 0


        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs_sample = torch.repeat_interleave(inputs, sample_n, dim=0)
            # labels_sample = torch.repeat_interleave(labels, sample_n, dim=0)
            inputs_sample = inputs_sample + torch.normal(mean=0.0, std=std, size=inputs_sample.shape).to(device)
            inputs_sample = inputs_sample.clamp(lower, upper)

            with torch.no_grad():
                outputs = model(inputs_sample)
            _, predicted = torch.max(outputs, 1)
        # 每sample_n个样本，取最多的那个为预测结果
            predicted = predicted.reshape(-1,sample_n)
            predicted = predicted.mode(dim = 1)[0]
            correct = (predicted == labels).sum().item()
            correct_sum += correct
        # construct the text file
        
        with open(f'/home/chizm/AdvRepair/results/sa1.txt','a') as f:
            f.write(f'model:{model_t}, radius:{radius}, dataloader: {loader} sample_n:{sample_n}, std:{std}, correct_rate:{correct_sum/(len(dataloader)*64)}\n')    
        print(f'model:{model_t}, radius:{radius}, dataloader: {loader} sample_n:{sample_n}, std:{std}, correct_rate:',correct_sum/(len(dataloader)*64))

    # sample(sample_n=sample_n,std=std,loader = 'testloader')
    # sample(sample_n=sample_n,std=std,loader = 'repairloader',dataloader = repairloader)
    # sample(sample_n=sample_n,std=std,loader = 'general_repairloader',dataloader = general_repairloader)
    sample(sample_n=sample_n,std=std,loader = 'adv_testloader',dataloader = adv_testloader)



def random_sample(sample_n,std,inputs, model,lower= 0.,upper = 1.,
                  device = 'cuda:0'):

    inputs = inputs.to(device)
    # labels = labels.to(device)
    inputs_sample = torch.repeat_interleave(inputs, sample_n, dim=0)
    inputs_sample = inputs_sample + torch.normal(mean=0.0, std=std, size=inputs_sample.shape).to(device)
    inputs_sample = inputs_sample.clamp(lower, upper)
    with torch.no_grad():
        outputs = model(inputs_sample)
    # top2
    _, predicted_2 = torch.topk(outputs, 2, 1, largest=True, sorted=True)
    champion = predicted_2[:,0]
    runner_up = predicted_2[:,1]
    champion = champion
    champion = champion.reshape(-1,sample_n)
    champion = champion.mode(dim = 1)[0]


    runner_up = runner_up.reshape(-1,sample_n)
    runner_up = runner_up.mode(dim = 1)[0]
    return champion, runner_up





def random_sample_dataloader(sample_n,std,dataloader, model,lower= 0.,upper = 1.,
                  device = 'cuda:0'):
    # sample the data
    '''
    return the predicted label,the runner_up label and the correct rate
    '''
    correct_sum = 0

    predicted_list = []
    runner_up_list = []
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        predicted, runner_up = random_sample(sample_n,std,inputs, model,
                                             lower = lower, upper = upper,
                                             device = device)
                                             

        predicted_list.append(predicted)
        correct = (predicted == labels).sum().item()
        correct_sum += correct


        runner_up_list.append(runner_up)

    predicted_list = torch.cat(predicted_list)
    runner_up_list = torch.cat(runner_up_list)
    return predicted_list, runner_up_list, correct_sum/(len(dataloader)*dataloader.batch_size)

    # sampling number 
    # cifar10
    # vgg19, std 0.2 sample_n: 20, testset: 0.90; repairset: 0.708; general_repairset: 0.6988
    # resnet18, rad 4, std 0.3 sample_n: 20, testset: 0.837; repairset: 0.615; general_repairset: 0.571
    # resnet18, rad 8, std 0.4 sample_n: 20, testset: 0.8235; repairset: 0.643; general_repairset: 0.593
    # mnist



    # tinyimagenet
    # wide_resnet101_2, rad 2, std 0.05 sample_n: 10, testset: 0.61; repairset: 0.53; general_repairset: 0.58
    # wide_resnet101_2, rad 2, std 0.1 sample_n: 10, testset: 0.48; repairset: 0.63; general_repairset: 0.64
    # wide_resnet101_2, rad 4, std 0.1 sample_n: 10, testset: 0.48; repairset: 0.38; general_repairset: 0.40
    # resnet152, rad 2, std 0.05 sample_n: 10, testset: 0.64; repairset: 0.47; general_repairset: 0.51
    # resnet152, rad 4, std 0.1 sample_n: 10, testset: 0.51; repairset: 0.45; general_repairset: 0.48

if __name__ == '__main__':
    # for std in [0.05,0.1,0.15,0.2,0.25,0.3]:
    # for model in ['FNN_small','FNN_big','CNN_small']:
    # for model in ['vgg19','resnet18']:
    for model in ['wide_resnet101_2','resnet152']:
    # for model in ['resnet18']:
        # for std in [0.35]:
        # for radius in [0.05,0.1,0.3]:
        for radius in [2,4]:
        # for radius in [4,8]:
            for std in [0.1]: #
                for sample in [10]:
                    # test_set_sample(model_t = model,radius = radius, dataset = 'mnist', sample_n=sample,std= std)
                    set_sample(model_t = model,radius = radius, dataset = 'tinyimagenet', sample_n=sample,std= std)
                    # set_sample(model_t = model,radius = radius, dataset = 'cifar10', sample_n=sample,std= std)
            # generate_adv_sample(model_t = model,radius = radius, dataset = 'tinyimagenet')
            # generate_adv_sample(model_t = model,radius = radius, dataset = 'cifar10')
    

    # for model in ['FNN_small','FNN_big','CNN_small']:
    #     for radius in [0.05,0.1,0.3]:
    #         generate_adv_sample(model_t = model,radius = radius, dataset = 'mnist')
