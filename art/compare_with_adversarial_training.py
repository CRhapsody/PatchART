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

MNIST_DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'MNIST' / 'processed'
MNIST_NET_DIR = Path(__file__).resolve().parent.parent / 'pgd' / 'model'

device = torch.device("cuda:2")

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
    def load(cls, train: bool, device):
        suffix = 'train' if train else 'test'
        fname = f'{suffix}_attack_data_full.pt'  # note that it is using original data
        # fname = f'{suffix}_norm00.pt'
        combine = torch.load(Path(MNIST_DATA_DIR, fname), device)
        inputs, labels = combine     
        # if train:
        # attack_data_fname = f'{suffix}_attack_data_part.pt'
        #     attack_combine = torch.load(Path(MNIST_DATA_DIR, attack_data_fname), device)
        #     attack_inputs, attack_labels = attack_combine
        #     inputs = torch.cat((inputs[:10000], attack_inputs), dim=0)
        #     labels = torch.cat((labels[:10000], attack_labels), dim=0)
        # if train:
        clean_data_fname = f'{suffix}_norm00.pt'
        clean_combine = torch.load(Path(MNIST_DATA_DIR, clean_data_fname), device)
        clean_inputs, clean_labels = clean_combine
        if train:
            # inputs = torch.cat((inputs[:10000], clean_inputs[:10000]), dim=0)
            # labels = torch.cat((labels[:10000], clean_labels[:10000]), dim=0)
            inputs = inputs[:10000]
            labels = labels[:10000]
        # else:
            # inputs = torch.cat((inputs[:2500]), dim=0)
            # labels = torch.cat((labels[:2500]), dim=0)
        
        assert len(inputs) == len(labels)
        return cls(inputs, labels)
    pass

def train():
    model = NeuralNet()
    
    model = model.to(device)
    model.load_state_dict(torch.load(Path(MNIST_NET_DIR, 'pdg_net.pth')))
    model.train()
    optimiser = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    train_data = MnistPoints.load(train=True, device=device)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    for epoch in range(20):
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

def test(model):
    model.eval()
    testset = MnistPoints.load(train=False, device=device)
    outs = model(testset.inputs)
    predicted = outs.argmax(dim=1)
    correct = (predicted == testset.labels).sum().item()
    ratio = correct / len(testset)

    return ratio

if __name__ == "__main__":
    model = train()
    ratio = test(model)
    print(ratio)
   