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

print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.version.cuda)


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
    x = self.fc2(x)
    x = torch.sigmoid(x)
    return x

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        
        # tensor1 = torch.tensor(item[0])  # 第一个张量 [60000]
        # tensor2 = torch.tensor(item[1])  # 第二个张量 [60000, 28, 28]

        # 在此可以执行其他的数据预处理操作

        return item[0], item[1]
  

if __name__ == "__main__":
    model = NeuralNet().to(device)
    model.load_state_dict(torch.load("/home/chizm/PatchART/pgd/model/pdg_net.pth"))
    model.eval()

    train = datasets.MNIST('./data/', train=True,
                       transform=transforms.Compose([transforms.ToTensor(),]),
                       download=False)
    
    test = datasets.MNIST('./data/', train=False, transform=transforms.Compose([transforms.ToTensor(),]),download=False)

    train_loader = DataLoader(train, batch_size=32)
    iter_train = iter(train_loader)
    # atk_images, atk_labels = iter_train.next()

    test_loader = DataLoader(test, batch_size=16)
    iter_test = iter(test_loader)
    # atk_images, atk_labels = iter_test.next()


    # train_set = torch.load('/home/chizm/PatchART/pgd/data/MNIST/processed/training.pt',map_location=device)
    # test_set = torch.load('/home/chizm/PatchART/pgd/data/MNIST/processed/test.pt',map_location=device)
    # print(train_set[0].shape,train_set[1].shape)
    # print(test_set[0].shape,test_set[1].shape)
    import math
    train_nbatch = math.ceil(60000/128)
    test_nbatch = math.ceil(10000/64)


    # custom_train_set = CustomDataset(train_set)
    # custom_test_set = CustomDataset(test_set)

    # train_DataLoader = DataLoader(custom_train_set,batch_size=32,shuffle=True)
    # test_DataLoader = DataLoader(custom_test_set,batch_size=16,shuffle=True)

    

    # train_DataLoader = iter(train_DataLoader)
    # test_DataLoader = iter(test_DataLoader)

    train_attacked_data = []
    train_labels = []
    test_attacked_data = []
    test_labels = []

    pgd = PGD(model=model, eps=0.3, alpha=2/255, steps=40, random_start=True)
    for i in range(train_nbatch):
        images,labels = iter_train.next()
        images = images.to(device)
        labels = labels.to(device)
        adv_images = pgd.forward(images,labels)
        outputs = model(adv_images)
        _, predicted = torch.max(outputs.data, 1)
        if torch.all(labels != predicted):
            print(f"train attack success {i}")
            train_attacked_data.append(adv_images)
            train_labels.append(labels)
    train_attack_data = torch.cat(train_attacked_data)
    train_attack_labels = torch.cat(train_labels)

    torch.save((train_attack_data,train_attack_labels),'./data/MNIST/processed/train_attack_data_full.pt')
    torch.save((train_attack_data[:1000],train_attack_labels[:1000]),'./data/MNIST/processed/train_attack_data_part.pt')
    for i in range(test_nbatch):
        images,labels = iter_test.next()
        images = images.to(device)
        labels = labels.to(device)
        adv_images = pgd.forward(images,labels)
        outputs = model(adv_images)
        _, predicted = torch.max(outputs.data, 1)
        if torch.all(labels != predicted):
            print(f"test attack success {i}")
            test_attacked_data.append(adv_images)
            test_labels.append(labels)
    test_attack_data = torch.cat(test_attacked_data)
    test_attack_labels = torch.cat(test_labels)

    torch.save((test_attack_data,test_attack_labels),'./data/MNIST/processed/test_attack_data_full.pt')
    torch.save((test_attack_data[:500],test_attack_labels[:500]),'./data/MNIST/processed/test_attack_data_part.pt')



