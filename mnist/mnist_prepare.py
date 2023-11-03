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
from pathlib import Path

py_file_location = "/home/chizm/PatchART/pgd"
sys.path.append(os.path.abspath(py_file_location))
sys.path.append(str(Path(__file__).resolve().parent.parent))

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
# print(f'Using {device} device')

# cuda prepare
# print(torch.cuda.is_available())
# print(torch.cuda.current_device())
# print(torch.version.cuda)


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet,self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        # self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3136, 100)
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
    def __init__(self,model,eps=0.3,alpha=3/255,steps=40,random_start=True):
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
    
    def forward_get_multi_datas(self,images,labels,number=5):
        images = images.clone().detach()
        labels = labels.clone().detach()


        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        attack_datas = []
        num = 0
        step = 0
        while True:
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            cost = loss(outputs, labels)
            grad = torch.autograd.grad(cost, adv_images,retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            # is successful?
            outputs = self.model(adv_images)
            _, predicted = torch.max(outputs.data, 1)
            if torch.all(labels != predicted):
                print(f"attack success {num}")
                attack_datas.append(adv_images)
                num+=1
            if step <= 200:
                step+=1
                print(f'process {step}')
            else:
                break
            if num < number:
                continue
            else:
                print(f"already collect {num} attacked data")
                # cat the attacked data
                adv_images_cat = torch.cat(attack_datas)
                # check every data is distinct
                adv_images_cat = torch.unique(adv_images_cat, dim=0)
                if adv_images_cat.size(0) >= number:
                    break
                else:
                    print(f"{adv_images_cat.size(0)} datas are not enough, continue to attack")
                    continue

        if attack_datas != []:
            return adv_images_cat
        else:
            return None
    
    def forward_sumsteps(self, images, labels, bitmap = None):
        images = images.clone().detach()
        labels = labels.clone().detach()


        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        steps = 0
        acc_flag = 0
        for step in range(self.steps):
            adv_images.requires_grad = True
            if bitmap is not None:
                in_lb, in_ub, in_bitmap = bitmap
                adv_bitmap = get_bitmap(in_lb, in_ub, in_bitmap, adv_images)
                outputs = self.model(adv_images, adv_bitmap)
            else:
                outputs = self.model(adv_images)
            cost = loss(outputs, labels)
            grad = torch.autograd.grad(cost, adv_images,retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

            steps+=1
            # is successful?
            if bitmap is not None:
                adv_bitmap = get_bitmap(in_lb, in_ub, in_bitmap, adv_images)
                outputs = self.model(adv_images, adv_bitmap)
            else:
                outputs = self.model(adv_images)
            _, predicted = torch.max(outputs.data, 1)
            if labels != predicted:
                print(f"attack success {steps}")
                acc_flag = 1
                return steps, acc_flag
        
        if steps == self.steps:
            print(f"attack fail {steps}")

        return steps, acc_flag

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

    
    input_data = torch.randn(320, 3, 128, 128).to('cuda:7')  # 8张128x128大小的彩色图片

    
    model.to('cuda:1')

    
    while(1):
        output = model(input_data)

        
        # print(output)

        
        # print(torch.cuda.max_memory_allocated() / 1e9, "GB")

def pgd_get_data(radius = 0.1, multi_number = 10, data_num = 200):
    '''
    pgd attack to origin data in radius, then get the five distinct attacked data from one origin data
    '''
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    model = NeuralNet().to(device)
    model.load_state_dict(torch.load("/home/chizm/PatchART/model/mnist/mnist.pth"))
    model.eval()
    # pgd attack
    train = datasets.MNIST('./data/', train=True,
                       transform=transforms.Compose([transforms.ToTensor(),]),
                       download=False)
    train_loader = DataLoader(train, batch_size=1)
    train_attacked_data = []
    train_labels = []
    train_attacked = []
    origin_data = []
    origin_label = []
    pgd = PGD(model=model, eps=radius, alpha=2/255, steps=10, random_start=True)
    i = 0
    for images,labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        adv_images = pgd.forward_get_multi_datas(images,labels,number=multi_number+1)
        if adv_images is not None and adv_images.size(0) >= multi_number+1:
            i += 1
        else:
            continue
        origin_data.append(images)
        origin_label.append(labels)

        train_attacked_data.append(adv_images)
        multi_labels = labels.unsqueeze(1).expand(-1,multi_number+1)
        train_labels.append(multi_labels)
        if i >= data_num:
            break
    origin_data = torch.cat(origin_data)
    origin_label = torch.cat(origin_label).reshape(-1)
    train_attack_data = torch.cat(train_attacked_data)
    train_attack_labels = torch.cat(train_labels).reshape(-1)
    # choose the first data from every multi_number+1 data,like 0,11,22,33,44,55,66,77,88,99 ...
    # then delete the train_repair_data from the train_attack_data
    train_repair_data = train_attack_data[multi_number::multi_number+1]
    train_repair_labels = train_attack_labels[multi_number::multi_number+1]
    data_mask = torch.ones_like(train_attack_data,dtype=torch.bool)
    data_mask[multi_number::multi_number+1] = False
    train_attack_data = train_attack_data[data_mask].reshape(-1,1,28,28)

    labels_mask = torch.ones_like(train_attack_labels,dtype=torch.bool)
    labels_mask[multi_number::multi_number+1] = False
    train_attack_labels = train_attack_labels[labels_mask]

    torch.save((origin_data,origin_label),f'./data/MNIST/processed/origin_data_{radius}_{data_num}.pt')
    torch.save((train_repair_data,train_repair_labels),f'./data/MNIST/processed/train_attack_data_full_{radius}_{data_num}.pt')
    torch.save((train_attack_data,train_attack_labels),f'./data/MNIST/processed/test_attack_data_full_{radius}_{data_num}.pt')



def grad_none(radius,data_num):
    # load
    origin_data,origin_label = torch.load(f'./data/MNIST/processed/origin_data_{radius}_{data_num}.pt')
    train_attack_data,train_attack_labels = torch.load(f'./data/MNIST/processed/train_attack_data_full_{radius}_{data_num}.pt')
    test_attack_data,test_attack_labels = torch.load(f'./data/MNIST/processed/test_attack_data_full_{radius}_{data_num}.pt')
    # grad none
    origin_data.requires_grad = False
    origin_label.requires_grad = False
    train_attack_data.requires_grad = False
    train_attack_labels.requires_grad = False
    test_attack_data.requires_grad = False
    test_attack_labels.requires_grad = False
    # save
    torch.save((origin_data,origin_label),f'./data/MNIST/processed/origin_data_{radius}.pt')
    torch.save((train_attack_data,train_attack_labels),f'./data/MNIST/processed/train_attack_data_full_{radius}.pt')
    torch.save((test_attack_data,test_attack_labels),f'./data/MNIST/processed/test_attack_data_full_{radius}.pt')

def get_trainset_norm00():
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    train = datasets.MNIST('/home/chizm/PatchART/data/', train=True,
                        transform=transforms.Compose([transforms.ToTensor(),]),
                        download=True)
    train_loader = DataLoader(train, batch_size=128)
    trainset_inputs = []
    trainset_labels = []
    for i,data in enumerate(train_loader,0):
        # collect batch of data and labels, then save as a tuple
        images, labels = data
        trainset_inputs.append(images)
        trainset_labels.append(labels)
        # print(f"batch {i} done")
    trainset_inputs = torch.cat(trainset_inputs)
    trainset_labels = torch.cat(trainset_labels)
    trainset_inputs.requires_grad = False
    trainset_labels.requires_grad = False
    torch.save((trainset_inputs[:10000],trainset_labels[:10000]),'/home/chizm/PatchART/data/MNIST/processed/train_norm00.pt')
    # 但它太大了，有180M

def adv_training(radius,data_num):
    # device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    model = NeuralNet().to(device)
    model.load_state_dict(torch.load("/home/chizm/PatchART/model/mnist/mnist.pth"))
    train_attack_data,train_attack_labels = torch.load(f'./data/MNIST/processed/train_attack_data_full_{radius}_{data_num}.pt',map_location=device)
    test_attack_data,test_attack_labels = torch.load(f'./data/MNIST/processed/test_attack_data_full_{radius}_{data_num}.pt',map_location=device)
    # dataset
    train_attack_dataset = torch.utils.data.TensorDataset(train_attack_data,train_attack_labels)
    test_attack_dataset = torch.utils.data.TensorDataset(test_attack_data,test_attack_labels)
    # data loader
    train_attack_loader = DataLoader(train_attack_dataset, batch_size=50)
    test_attack_loader = DataLoader(test_attack_dataset, batch_size=128)
    # train
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(25):
        print(f"epoch {epoch}")
        epoch_loss = 0
        correct, total = 0,0
        for inputs,labels in train_attack_loader:

        # for i in range(train_nbatch):
        #     inputs,labels = iter_train.__next__()
            inputs = inputs.to(device)
            labels = labels.to(device)
            for step in range(50):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
    
    return model.eval()

def adv_training_test(radius):
    # load net
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    model = NeuralNet().to(device)
    model.load_state_dict(torch.load("/home/chizm/PatchART/model/mnist/mnist.pth"))

    # load attack data
    train_attack_data,train_attack_labels = torch.load(f'./data/MNIST/processed/train_attack_data_full_{radius}.pt')
    test_attack_data,test_attack_labels = torch.load(f'./data/MNIST/processed/test_attack_data_full_{radius}.pt')
    test_data, test_labels = torch.load('./data/MNIST/processed/test_norm00.pt')
    train_data,train_labels = torch.load('/pub/data/chizm/train_norm00.pt')
    print(torch.cuda.max_memory_allocated() / 1e9, "GB")
    # dataset
    train_attack_dataset = torch.utils.data.TensorDataset(train_attack_data,train_attack_labels)
    test_attack_dataset = torch.utils.data.TensorDataset(test_attack_data,test_attack_labels)
    train_dataset = torch.utils.data.TensorDataset(train_data,train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_data,test_labels)
    # data loader
    train_attack_loader = DataLoader(train_attack_dataset, batch_size=50)
    test_attack_loader = DataLoader(test_attack_dataset, batch_size=128)
    train_loader = DataLoader(train_dataset, batch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=128)
    # train
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(25):
        print(f"epoch {epoch}")
        epoch_loss = 0
        correct, total = 0,0
        for inputs,labels in train_attack_loader:

        # for i in range(train_nbatch):
        #     inputs,labels = iter_train.__next__()
            inputs = inputs.to(device)
            labels = labels.to(device)
            for step in range(50):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                # pred = torch.max(outputs,1)
                # total += labels.size(0)
                # correct += (pred.indices == labels).sum().item()
        #         pred = torch.max(outputs,1)
        #         acc = (pred == labels).sum().item()/labels.size(0)
        #     print(" Loss: ",epoch_loss," Accuracy:",acc)
        # print("epoch:",epoch+1, " Loss: ",epoch_loss," Accuracy:",acc)
        
        
    # test
    model.eval()
    total = 0
    train_attack_correct = 0
    for data, target in train_attack_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        total += target.size(0)
        train_attack_correct += pred.eq(target.view_as(pred)).sum().item()
    print(f"train_attack_loader {train_attack_correct/total}")
    
    # test
    model.eval()
    total = 0
    test_attack_correct = 0
    for data, target in test_attack_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        total += target.size(0)
        test_attack_correct += pred.eq(target.view_as(pred)).sum().item()
    print(f"test_attack_loader {test_attack_correct/total}")
    
    train_correct = 0
    total = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        total += target.size(0)
        train_correct += pred.eq(target.view_as(pred)).sum().item()
    print(f"train_loader {train_correct/total}")

    test_correct = 0
    total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        total += target.size(0)
        test_correct += pred.eq(target.view_as(pred)).sum().item()
    print(f"test_loader {test_correct/total}")

# judge the batch_inputs is in which region of property
from torch import Tensor
def get_bitmap(in_lb: Tensor, in_ub: Tensor, in_bitmap: Tensor, batch_inputs: Tensor):
    '''
    in_lb: n_prop * input
    in_ub: n_prop * input
    batch_inputs: batch * input
    '''
    with torch.no_grad():
    
        batch_inputs_clone = batch_inputs.clone().unsqueeze_(1)
        # distingush the photo and the property
        if len(in_lb.shape) == 2:
            batch_inputs_clone = batch_inputs_clone.expand(batch_inputs.shape[0], in_lb.shape[0], in_lb.shape[1])
        elif len(in_lb.shape) == 4:
            batch_inputs_clone = batch_inputs_clone.expand(batch_inputs.shape[0], in_lb.shape[0], in_lb.shape[1], in_lb.shape[2], in_lb.shape[3])
        is_in = (batch_inputs_clone >= in_lb) & (batch_inputs_clone <= in_ub)
        if len(in_lb.shape) == 2:
            is_in = is_in.all(dim=-1) # every input is in the region of property, batch * n_prop
        elif len(in_lb.shape) == 4:
            is_in = is_in.all(dim=(-1)).all(dim=(-1)).all(dim=(-1)) # every input is in the region of property, batch * n_prop
        # convert to bitmap
        bitmap = torch.zeros((batch_inputs.shape[0], in_bitmap.shape[1]), device = device)

        for i in range(is_in.shape[0]):
            for j in range(is_in.shape[1]):
                if is_in[i][j]:
                    bitmap[i] = in_bitmap[j]
                    break
                else:
                    continue

        return bitmap

def compare_pgd_step_length(radius, repair_number):
    '''
    use the length of pgd steps to compare the hardness of attacking two model respectively
    the model1 is origin model, model2 is repaired model
    '''
    # load net
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    model1 = NeuralNet().to(device)
    model1.load_state_dict(torch.load("/home/chizm/PatchART/model/mnist/mnist.pth"))

    from mnist.mnist_utils import MnistNet, Mnist_patch_model,MnistProp
    from art.repair_moudle import Netsum
    from DiffAbs.DiffAbs import deeppoly

    net = MnistNet(dom=deeppoly)
    net.to(device)
    patch_lists = []
    for i in range(repair_number):
        patch_net = Mnist_patch_model(dom=deeppoly,
            name = f'patch network {i}')
        patch_net.to(device)
        patch_lists.append(patch_net)
    model2 =  Netsum(deeppoly, target_net = net, patch_nets= patch_lists, device=device)
    model2.load_state_dict(torch.load(f"/home/chizm/PatchART/model/reassure_format/Mnist-repair_number{repair_number}-rapair_radius{radius}-.pt",map_location=device))

    model3 = adv_training(radius, data_num=repair_number)


    # load data
    datas,labels = torch.load(f'/home/chizm/PatchART/data/MNIST/processed/origin_data_{radius}_{repair_number}.pt',map_location=device)
    # return
    
    # datas = datas[:repair_number]
    # labels = labels[:repair_number]

    # pgd
    pgd1 = PGD(model=model1, eps=radius, alpha=2/255, steps=200, random_start=True)
    pgd2 = PGD(model=model2, eps=radius, alpha=2/255, steps=200, random_start=True)
    pgd3 = PGD(model=model3, eps=radius, alpha=2/255, steps=200, random_start=True)

    # attack
    ori_step = 0
    repair_step = 0
    pgd_step = 0

    # get bitmap
    from art.prop import AndProp
    from art.bisecter import Bisecter
    repairlist = [(data[0],data[1]) for data in zip(datas, labels)]
    repair_prop_list = MnistProp.all_props(deeppoly, DataList=repairlist, input_shape= datas.shape[1:], radius= radius)
    # get the all props after join all l_0 ball feature property
    # TODO squeeze the property list, which is the same as the number of label
    all_props = AndProp(props=repair_prop_list)
    v = Bisecter(deeppoly, all_props)
    in_lb, in_ub = all_props.lbub(device)
    in_bitmap = all_props.bitmap(device)

    p1 = 0
    p2 = 0
    p3 = 0

    for image, label in zip(datas,labels):
        image = image.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)
        
        step1, ori_acc = pgd1.forward_sumsteps(image,label)
        step2, repair_acc = pgd2.forward_sumsteps(image,label,bitmap = [in_lb, in_ub, in_bitmap])
        step3, adt_acc = pgd3.forward_sumsteps(image,label)
        ori_step += step1
        repair_step += step2
        pgd_step += step3
        if ori_acc == 1:
            p1 += 1
        if repair_acc == 1:
            p2 += 1
        if adt_acc == 1:
            p3 += 1
            
    
    print(f"ori_step {ori_step}, repair_step {repair_step}, pgd_step {pgd_step} \\ p1:{p1}, p2:{p2}, p3:{p3}")
    with open(f'./data/MNIST/processed/compare_pgd_step_length_{radius}_{repair_number}.txt','w') as f:
        f.write(f"ori_step {ori_step}, repair_step {repair_step}, pgd_step {pgd_step} \\ p1:{p1}, p2:{p2}, p3:{p3}")



if __name__ == "__main__":
    pass
    # model = train()
    # test()
    # pgd_attack()
    # stack()
    compare_pgd_step_length(radius=0.1,repair_number=1000)

    for data in [500,1000]:
        for radius in [0.3]:
            # pgd_get_data(radius=radius,multi_number=10,data_num=data)
    # pgd_get_data(radius=0.3,multi_number=10,data_num=1000)
                # grad_none(radius, data_num=data)
    # get_trainset_norm00()
            compare_pgd_step_length(radius=radius,repair_number=data)

