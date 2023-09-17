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

py_file_location = "/home/chizm/PatchART/pgd"
sys.path.append(os.path.abspath(py_file_location))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# cuda prepare
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.version.cuda)


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
    
    # split the model into two parts, first part is the feature extractor until fc1, second part is the classifier
    def split(self):
        return nn.Sequential(
            self.conv1,
            self.maxpool,
            self.relu,
            self.conv2,
            self.maxpool,
            self.relu,
            # torch.flatten(x, 1),
            nn.Flatten(),
            self.fc1,
            self.relu
        ), nn.Sequential(
            
            self.fc2
            # nn.Sigmoid()
        )
    
    # use the self.split() to get the feature extractor until fc1
    def get_the_feature(self,x):
        x = self.split()[0](x)
        return x

class PGD():
  def __init__(self,model,eps=0.3,alpha=2/255,steps=40,random_start=True):
    self.eps = eps
    self.model = model
    self.attack = "Projected Gradient Descent"
    self.alpha = alpha
    self.steps = steps
    self.random_start = random_start
    self.supported_mode = ["default"]
  
  def forward(self,images,labels):
    images = images.clone().detach()
    labels = labels.clone().detach()


    loss = nn.CrossEntropyLoss()

    adv_images = images.clone().detach()

    if self.random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for step in range(self.steps):
        adv_images.requires_grad = True
        outputs = self.model(adv_images)
        cost = loss(outputs, labels)
        grad = torch.autograd.grad(cost, adv_images,retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + self.alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    return adv_images

import math
def train():
    train = datasets.MNIST('./data/', train=True,
                       transform=transforms.Compose([transforms.ToTensor(),]),
                       download=True)
    train_loader = DataLoader(train, batch_size=128)
    iter_train = iter(train_loader)
    train_nbatch = math.ceil(60000/128)
    model = NeuralNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(10):
        epoch_loss = 0
        correct, total = 0,0
        for i in range(train_nbatch):
            inputs,labels = iter_train.__next__()
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pred = torch.max(outputs,1)
            total += labels.size(0)
            correct += (pred.indices == labels).sum().item()
        print("Epoch:",epoch+1, " Loss: ",epoch_loss," Accuracy:",correct/total)
        
    torch.save(model.state_dict(), './model/mnist.pth')
    return model


def test(model):
    # test
    test = datasets.MNIST('./data/', train=False,
                      transform=transforms.Compose([transforms.ToTensor(),]),
                      download=True)
    test_loader = DataLoader(test, batch_size=128)
    model.eval()
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == "__main__":
    pass
    # model = train()
    # test(model)
