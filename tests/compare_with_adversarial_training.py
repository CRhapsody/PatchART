import sys
import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms, datasets
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
# import matplotlib.pyplot as plt
import numpy as np
import exp
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets


MNIST_DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'MNIST' / 'processed'
MNIST_NET_DIR = Path(__file__).resolve().parent.parent / 'pgd' / 'model'

device = torch.device("cuda:0")

def get_trainset(number):
    train = datasets.MNIST('./data/', train=True,
                       transform=transforms.Compose([transforms.ToTensor()]),
                       download=True)
    train_loader = DataLoader(train, batch_size=128)
    
    traindata_list = []
    trainlabel_list = []
    for x,y in train_loader:
        traindata_list.append(x)
        trainlabel_list.append(y)
    train_data = torch.cat(traindata_list)
    train_label = torch.cat(trainlabel_list)
    torch.save((train_data[:number],train_label[:number]),"/home/chizm/PatchART/data/MNIST/processed/train_norm00.pt")

class NeuralNet(nn.Module):
  def __init__(self):
      super(NeuralNet,self).__init__()
      self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
      self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
      self.maxpool = nn.MaxPool2d(2)
      self.relu = nn.ReLU()
      self.fc1 = nn.Linear(1024, 32)
      self.fc2 = nn.Linear(32, 10)

  def forward(self,x):
      x = self.conv1(x)
      x = self.maxpool(x)
      x = self.relu(x)
      x = self.conv2(x)
      x = self.maxpool(x)
      x = self.relu(x)
      # x = torch.flatten(x, 1)
      x = x.view(-1,1024)
      x = self.fc1(x)
      x = self.relu(x)
      x = self.fc2(x)
      # x = torch.sigmoid(x)
      return x

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
                fname = f'{suffix}_norm00.pt'
                combine = torch.load(Path(MNIST_DATA_DIR, fname), device)
                inputs, labels = combine
                inputs = inputs[:testnumber]
                labels = labels[:testnumber]
            
            elif is_attack_testset_repaired:
                fname = f'{suffix}_attack_data_full.pt'
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

def train(trainnumber = None, original = False):
    model = NeuralNet()
    
    model = model.to(device)
    model.load_state_dict(torch.load(Path(MNIST_NET_DIR, 'pdg_net.pth')))
    if original:
        return model
    model.train()
    optimiser = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    train_data = MnistPoints.load(train=True, device=device,trainnumber=trainnumber)
    train_loader = DataLoader(train_data, batch_size=trainnumber, shuffle=True)
    for epoch in range(45):
        epoch_loss = 0
        correct, total = 0,0
        for i,data in enumerate(train_loader,0):
            inputs,labels = data
            optimiser.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs,labels)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
            pred = torch.max(outputs,1)
            total += labels.size(0)
            correct += (pred.indices == labels).sum().item()
        print("Epoch:",epoch+1, " Loss: ",epoch_loss," Accuracy:",correct/total)
    return model

def test(model,testnumber = None,
            is_test_accuracy = False,
            is_attack_testset_repaired = False,
            is_attack_repaired = False):
    model.eval()
    testset = MnistPoints.load(train=False, device=device, testnumber=testnumber,
                               is_test_accuracy = is_test_accuracy,
                               is_attack_testset_repaired = is_attack_testset_repaired,
                               is_attack_repaired = is_attack_repaired)
    outs = model(testset.inputs)
    predicted = outs.argmax(dim=1)
    correct = (predicted == testset.labels).sum().item()
    ratio = correct / len(testset)

    return ratio

if __name__ == "__main__":
    # get_trainset(10000)
    model = train(trainnumber=150 ,original=False)
    ratio = test(model, testnumber=2000, 
                 is_test_accuracy=True,
                 is_attack_testset_repaired=False,
                 is_attack_repaired=False)
    print(ratio)
   