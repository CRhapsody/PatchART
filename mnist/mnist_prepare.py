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
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        # self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16*14*14, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self,x):
        x = self.conv1(x)
        # x = self.maxpool(x)
        x = self.relu(x)
        # x = self.conv2(x)
        # x = self.maxpool(x)
        # x = self.relu(x)
        # x = torch.flatten(x, 1)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = torch.sigmoid(x)
        return x
    
    # split the model into two parts, first part is the feature extractor until fc1, second part is the classifier
    # def split(self):
    #     return nn.Sequential(
    #         self.conv1,
    #         self.maxpool,
    #         self.relu,
    #         self.conv2,
    #         self.maxpool,
    #         self.relu,
    #         # torch.flatten(x, 1),
    #         nn.Flatten(),
    #         self.fc1,
    #         self.relu
    #     ), nn.Sequential(
            
    #         self.fc2
    #         # nn.Sigmoid()
    #     )
    
    # # use the self.split() to get the feature extractor until fc1
    # def get_the_feature(self,x):
    #     x = self.split()[0](x)
    #     return x

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
    # iter_train = iter(train_loader)
    # train_nbatch = math.ceil(60000/128)
    model = NeuralNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(10):
        epoch_loss = 0
        correct, total = 0,0
        for inputs,labels in train_loader:
        # for i in range(train_nbatch):
        #     inputs,labels = iter_train.__next__()
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


def test():
    # test
    test = datasets.MNIST('./data/', train=False,
                      transform=transforms.Compose([transforms.ToTensor(),]),
                      download=True)
    test_loader = DataLoader(test, batch_size=128)
    model = NeuralNet().to(device)
    model.load_state_dict(torch.load('./model/mnist.pth'))
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

def pgd_attack():
    model = NeuralNet().to(device)
    model.load_state_dict(torch.load("/home/chizm/PatchART/model/mnist.pth"))
    model.eval()

    train = datasets.MNIST('./data/', train=True,
                       transform=transforms.Compose([transforms.ToTensor(),]),
                       download=False)
    
    test = datasets.MNIST('./data/', train=False, transform=transforms.Compose([transforms.ToTensor(),]),download=False)

    train_loader = DataLoader(train, batch_size=256)
    # atk_images, atk_labels = iter_train.next()
    test_loader = DataLoader(test, batch_size=64)
    train_attacked_data = []
    train_labels = []
    train_attacked = []
    test_attacked_data = []
    test_labels = []
    test_attacked = []
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    pgd = PGD(model=model, eps=0.03, alpha=2/255, steps=10, random_start=True)
    i = 0
    for images,labels in train_loader:
        i += 1
        images = images.to(device)
        labels = labels.to(device)
        adv_images = pgd.forward(images,labels)
        outputs = model(adv_images)
        _, predicted = torch.max(outputs.data, 1)
        if torch.all(labels != predicted):
            print(f"train attack success {i}")
            train_attacked_data.append(adv_images)
            train_labels.append(labels)
            train_attacked.append(predicted)
        else:
            train_attacked_data.append(adv_images[labels != predicted])
            train_labels.append(labels[labels != predicted])
            train_attacked.append(predicted[labels != predicted])

    train_attack_data = torch.cat(train_attacked_data)
    train_attack_labels = torch.cat(train_labels)
    train_attacked = torch.cat(train_attacked)

    with torch.no_grad():
        outs = model(train_attack_data)
        predicted = outs.argmax(dim=1)
        correct = (predicted == train_attack_labels).sum().item()
        ratio = correct / len(train_attack_data)

    torch.save((train_attack_data,train_attack_labels),'./data/MNIST/processed/train_attack_data_full.pt')
    torch.save((train_attack_data[:5000],train_attack_labels[:5000]),'./data/MNIST/processed/train_attack_data_part_5000.pt')
    torch.save(train_attacked[:5000],'./data/MNIST/processed/train_attack_data_part_label_5000.pt')

    pgd = PGD(model=model, eps=0.05, alpha=1/255, steps=100, random_start=True)
    i = 0
    for images,labels in test_loader:
        i += 1
        images = images.to(device)
        labels = labels.to(device)
        adv_images = pgd.forward(images,labels)
        outputs = model(adv_images)
        _, predicted = torch.max(outputs.data, 1)
        if torch.all(labels != predicted):
            print(f"test attack success {i}")
            test_attacked_data.append(adv_images)
            test_labels.append(labels)
            test_attacked.append(predicted)
        else:
            test_attacked_data.append(adv_images[labels != predicted])
            test_labels.append(labels[labels != predicted])
            test_attacked.append(predicted[labels != predicted])
    test_attack_data = torch.cat(test_attacked_data)
    test_attack_labels = torch.cat(test_labels)
    test_attacked = torch.cat(test_attacked)

    torch.save((test_attack_data,test_attack_labels),'./data/MNIST/processed/test_attack_data_full.pt')
    torch.save((test_attack_data[:2500],test_attack_labels[:2500]),'./data/MNIST/processed/test_attack_data_part_2500.pt')
    torch.save(test_attacked[:2500],'./data/MNIST/processed/test_attack_data_part_label_2500.pt')


def stack():

    # 定义一个深层卷积神经网络
    class DeepCNN(nn.Module):
        def __init__(self):
            super(DeepCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(256 * 64 * 64, 512)
            self.fc2 = nn.Linear(512, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = x.view(-1, 256 * 64 * 64)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    
    model = DeepCNN()

    
    input_data = torch.randn(320, 3, 128, 128).to('cuda')  # 8张128x128大小的彩色图片

    
    model.to('cuda')

    
    while(1):
        output = model(input_data)

        
        # print(output)

        
        # print(torch.cuda.max_memory_allocated() / 1e9, "GB")



if __name__ == "__main__":
    pass
    # model = train()
    # test()
    # pgd_attack()
    stack()
