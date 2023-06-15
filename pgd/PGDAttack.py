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
            self.fc1
        ), nn.Sequential(
            
            self.fc2,
            nn.Sigmoid()
        )
    
    # use the self.split() to get the feature extractor until fc1
    def get_the_feature(self,x):
        x = self.split()[0](x)
        return x

def cluster(device: str):
    model = NeuralNet().to(device)
    model.load_state_dict(torch.load("/home/chizm/PatchART/pgd/model/pdg_net.pth"))

    # load the train data and test data from the pt file respectively
    train_data,train_label = torch.load('/home/chizm/PatchART/data/MNIST/processed/train_attack_data_part.pt',map_location=device)
    test_data,test_label = torch.load('/home/chizm/PatchART/data/MNIST/processed/test_attack_data_part.pt',map_location=device)
    train_attack_label = torch.load('/home/chizm/PatchART/data/MNIST/processed/train_attack_data_part_label.pt',map_location=device)
    test_attack_label = torch.load('/home/chizm/PatchART/data/MNIST/processed/test_attack_data_part_label.pt',map_location=device)

    # unsqueeze the label of train data and test data to 4 dimensions
    # train_label = train_label.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    # test_label = test_label.unsqueeze(1).unsqueeze(2).unsqueeze(3)

    # combine the train data and train label, test data and test label respectively
    train_set = torch.utils.data.TensorDataset(train_data,train_label)
    test_set = torch.utils.data.TensorDataset(test_data,test_label)

    # load the train data and test data to the dataloader respectively
    train_loader = DataLoader(train_set, batch_size=32)
    test_loader = DataLoader(test_set, batch_size=16)

    # iterate the train data and test data 
    iter_train = iter(train_loader)
    iter_test = iter(test_loader)

    # record the feature of train data and test data using list
    train_feature = []
    train_label = []
    test_feature = []
    test_label = []

    with torch.no_grad():
        model1, model2 = model.split()
        model1.eval()
        for i in range(len(train_loader)):
            images, labels = iter_train.next()
            images = images.to(device)
            labels = labels.to(device)
            feature = model1(images)
            train_feature.append(feature)
            train_label.append(labels)
        for i in range(len(test_loader)):
            images, labels = iter_test.next()
            images = images.to(device)
            labels = labels.to(device)
            feature = model1(images)
            test_feature.append(feature)
            test_label.append(labels)
    train_feature = torch.cat(train_feature, dim=0)
    train_label = torch.cat(train_label, dim=0)
    test_feature = torch.cat(test_feature, dim=0)
    test_label = torch.cat(test_label, dim=0)

    # use the kmeans to cluster the train feature and test feature respectively
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=10, random_state=0).fit(train_feature.cpu().numpy())
    train_cluster = kmeans.labels_
    kmeans = KMeans(n_clusters=10, random_state=0).fit(test_feature.cpu().numpy())
    test_cluster = kmeans.labels_

    # use the SpectralClustering to cluster the train feature and test feature respectively
    ######## failed
    # from sklearn.cluster import SpectralClustering
    # spectral = SpectralClustering(n_clusters=10, random_state=0).fit(train_feature.cpu().numpy())
    # train_cluster = spectral.labels_
    # spectral = SpectralClustering(n_clusters=10, random_state=0).fit(test_feature.cpu().numpy())
    # test_cluster = spectral.labels_

    # use the gmm to cluster the train feature and test feature respectively
    from sklearn.mixture import GaussianMixture
    # gmm = GaussianMixture(n_components=10, random_state=0).fit(train_feature.cpu().numpy())
    # train_cluster = gmm.predict(train_feature.cpu().numpy())
    # gmm = GaussianMixture(n_components=10, random_state=0).fit(test_feature.cpu().numpy())
    # test_cluster = gmm.predict(test_feature.cpu().numpy())

    # save the feature of train data and test data respectively
    torch.save(train_feature, '/home/chizm/PatchART/data/MNIST/processed/train_feature_part.pt')
    torch.save(test_feature, '/home/chizm/PatchART/data/MNIST/processed/test_feature_part.pt')

    # compare the cluster result with the label of train data and test data respectively
    train_cluster = torch.from_numpy(train_cluster)
    test_cluster = torch.from_numpy(test_cluster)
    train_cluster = train_cluster.to(device)
    test_cluster = test_cluster.to(device)
    train_label = train_label.to(device)
    test_label = test_label.to(device)
    train_correct = torch.sum(train_cluster == train_label)
    test_correct = torch.sum(test_cluster == test_label)
    print("train correct: ", train_correct)



            




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


def pgd():
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
    train_attacked = []
    test_attacked_data = []
    test_labels = []
    test_attacked = []
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    pgd = PGD(model=model, eps=0.1, alpha=2/255, steps=80, random_start=True)
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
            train_attacked.append(predicted)
        else:
            train_attacked_data.append(images[labels != predicted])
            train_labels.append(labels[labels != predicted])
            train_attacked.append(predicted[labels != predicted])

    train_attack_data = torch.cat(train_attacked_data)
    train_attack_labels = torch.cat(train_labels)
    train_attacked = torch.cat(train_attacked)

    torch.save((train_attack_data,train_attack_labels),'./data/MNIST/processed/train_attack_data_full.pt')
    torch.save((train_attack_data[:1000],train_attack_labels[:1000]),'./data/MNIST/processed/train_attack_data_part.pt')
    torch.save(train_attacked[:1000],'./data/MNIST/processed/train_attack_data_part_label.pt')

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
            test_attacked.append(predicted)
        else:
            test_attacked_data.append(images[labels != predicted])
            test_labels.append(labels[labels != predicted])
            test_attacked.append(predicted[labels != predicted])
    test_attack_data = torch.cat(test_attacked_data)
    test_attack_labels = torch.cat(test_labels)
    test_attacked = torch.cat(test_attacked)

    torch.save((test_attack_data,test_attack_labels),'./data/MNIST/processed/test_attack_data_full.pt')
    torch.save((test_attack_data[:500],test_attack_labels[:500]),'./data/MNIST/processed/test_attack_data_part.pt')
    torch.save(test_attacked[:500],'./data/MNIST/processed/test_attack_data_part_label.pt')


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cluster(device=device)
    # pgd()




