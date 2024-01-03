import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from vgg import VGG
# from resnet import ResNet18
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import OrderedDict
# import io
# Define the Vgg19 model
from torchvision.models import resnet18
from cifar10_utils import Resnet_model

import os
from pathlib import Path
import sys
py_file_location = "/home/chizm/PatchART/pgd"
sys.path.append(os.path.abspath(py_file_location))
sys.path.append(str(Path(__file__).resolve().parent.parent))

def training(model_type:str,device):
    '''
    model: str, the name of the model
    '''
    if model_type == 'vgg19':
        model = VGG('VGG19')
        # model.classifier[6] = nn.Linear(4096, 10)
        model.to(device)
    elif model_type == 'resnet18':
        model = ResNet18()
        model.to(device)


# split the last fc layer from vgg19
# model_last_fc = nn.Sequential(*list(vgg19_model.children())[-1:])




    # Load the CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize the FNN model

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # Train the FNN model
    num_epochs = 30
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()            

            running_loss += loss.item()
            if (i+1) % 49 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0
        scheduler.step()

    # Test the FNN model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')

    # Save the model
    torch.save(model.state_dict(), './model/cifar10/cifar10_vgg19.pth')

def trasfer_state_dict(model:str):
        state = torch.load(f"/home/chizm/PatchART/model/cifar10/{model}.pth")
        # need replace the module.xx to xx
        new_state_dict = OrderedDict()
        for key, value in state.items():
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = value
        torch.save(new_state_dict,f"/home/chizm/PatchART/model/cifar10/{model}.pth")


def test_dataset(model_type:str, dataset_type:str, radius:int, device):
    if model_type == 'vgg19':
        model = VGG('VGG19')
        # model.classifier[6] = nn.Linear(4096, 10)
        state = torch.load(f"/home/chizm/PatchART/model/cifar10/vgg19.pth")
        model.load_state_dict(state)
    elif model_type == 'resnet18':
        from torchvision.models import resnet18
        model = resnet18(num_classes=10)
        state = torch.load(f"/home/chizm/PatchART/model/cifar10/resnet18.pth")
        new_state_dict = OrderedDict()
        for key, value in state.items():
            new_key = key.replace('module.', '')  # 去掉'module.'前缀
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict)

    model.to(device)

    # Load the CIFAR-10 dataset
    if dataset_type == 'repair':
        dataset = torch.load(f'/home/chizm/PatchART/data/cifar10/train_attack_data_full_{model_type}_{radius}.pt')
    elif dataset_type == 'origin':
        dataset = torch.load(f'/home/chizm/PatchART/data/cifar10/origin_data_{model_type}_{radius}.pt')
    elif dataset_type == 'attack_test':
        dataset = torch.load(f'/home/chizm/PatchART/data/cifar10/test_attack_data_full_{model_type}_{radius}.pt')
    elif dataset_type == 'test':
        dataset = torch.load(f'/home/chizm/PatchART/data/cifar10/test_norm.pt')
    elif dataset_type == 'train':
        dataset = torch.load(f'/home/chizm/PatchART/data/cifar10/train_norm.pt')

    data, label = dataset

    model.eval()
    # model.train(False)
    data.to(device)
    label.to(device)
    with torch.no_grad():
        # correct = 0
        # total = 0
        # for i in range(data.size(0)):
        #     images = data[i].unsqueeze(0).to(device)
        #     labels = label[i].unsqueeze(0).to(device)
        #     outputs = model(images)
        #     _, predicted = torch.max(outputs.data, 1)
        #     total += labels.size(0)
        #     correct += (predicted == labels).sum().item()

        # accuracy = 100 * correct / total
        # print(f'{dataset_type} Test Accuracy: {accuracy:.2f}%')



        outputs = model(data)
        predicted = outputs.argmax(dim=1)
        correct = (predicted == label).sum().item()
        # ratio = correct / len(testset)
        ratio = correct / len(data)
        print(f'{dataset_type} Test Accuracy: {ratio}')


def get_dataset():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train = datasets.CIFAR10('/home/chizm/PatchART/data/', train=True,
                        transform=transform,
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
    trainset_inputs_split = trainset_inputs[:10000]
    trainset_labels_split = trainset_labels[:10000]
    trainset_inputs_split.requires_grad = False
    trainset_labels_split.requires_grad = False
    torch.save((trainset_inputs_split, trainset_labels_split),'/home/chizm/PatchART/data/cifar10/train.pt')
    # torch.save((trainset_inputs,trainset_labels),'/home/chizm/PatchART/data/cifar10/train_norm00_full.pt')
    test = datasets.CIFAR10('/home/chizm/PatchART/data/', train=False,
                        transform=transform,
                        download=True)
    test_loader = DataLoader(test, batch_size=128)

    testset_inputs = []
    testset_labels = []
    for i,data in enumerate(test_loader,0):
        # collect batch of data and labels, then save as a tuple
        images, labels = data
        testset_inputs.append(images)
        testset_labels.append(labels)
        # print(f"batch {i} done")
    testset_inputs = torch.cat(testset_inputs)
    testset_labels = torch.cat(testset_labels)
    testset_inputs.requires_grad = False
    testset_labels.requires_grad = False
    torch.save((testset_inputs,testset_labels),'/home/chizm/PatchART/data/cifar10/test.pt')

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
            # adv_images = torch.clamp(adv_images, min=0, max=1).detach()

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
            # adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            adv_images = images + delta
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
                continue
            
            else:
                print(f"already collect {num} attacked data")
                # cat the attacked data
                if attack_datas == []:
                    return None
                adv_images_cat = torch.cat(attack_datas)
                # check every data is distinct
                adv_images_cat = torch.unique(adv_images_cat, dim=0)
                if adv_images_cat.size(0) >= number:
                    adv_images_cat = adv_images_cat[:number]
                    break
                elif adv_images_cat.shape[0] < number:
                    return None
                else:
                    print(f"{adv_images_cat.size(0)} datas are not enough, continue to attack")
                    continue

        # if attack_datas != []:
        return adv_images_cat
            
    
    def forward_sumsteps(self, images, labels, device = None, bitmap = None):
        images = images.clone().detach()
        labels = labels.clone().detach()


        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            # adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        steps = 0
        acc_flag = 0
        for step in range(self.steps):
            adv_images.requires_grad = True
            if bitmap is not None:
                in_lb, in_ub, in_bitmap = bitmap
                adv_bitmap = get_bitmap(in_lb, in_ub, in_bitmap, adv_images, device)
                outputs = self.model(adv_images, adv_bitmap)
            else:
                outputs = self.model(adv_images)
            cost = loss(outputs, labels)
            grad = torch.autograd.grad(cost, adv_images,retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            # adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            adv_images = images + delta

            steps+=1
            # is successful?
            if bitmap is not None:
                adv_bitmap = get_bitmap(in_lb, in_ub, in_bitmap, adv_images, device)
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



def pgd_get_data(net, radius = 2, multi_number = 10, data_num = 200, general = False):
    '''
    pgd attack to origin data in radius, then get the five distinct attacked data from one origin data
    '''
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    if net == 'vgg19':
        model = VGG('VGG19').to(device)
        state = torch.load(f"/home/chizm/PatchART/model/cifar10/vgg19.pth")['net']
        # need replace the module.xx to xx
        new_state_dict = OrderedDict()
        for key, value in state.items():
            new_key = key.replace('module.', '')  # 去掉'module.'前缀
            new_state_dict[new_key] = value
        torch.save(new_state_dict,f"/home/chizm/PatchART/model/cifar10/vgg19.pth")

        model.load_state_dict(new_state_dict)

    elif net == 'resnet18':
        model = resnet18(num_classes=10).to(device)
        # from resnet import ResNet18
        # model = ResNet18().to(device)
        state = torch.load(f"/home/chizm/PatchART/model/cifar10/resnet18.pth")
        # need replace the module.xx to xx
        new_state_dict = OrderedDict()
        for key, value in state.items():
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = value
        # torch.save(new_state_dict,f"/home/chizm/PatchART/model/cifar10/resnet18.pth")
        model.load_state_dict(new_state_dict)
    model.eval()




    # pgd attack
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    if general == True:
        train = torch.utils.data.Subset(train,range(40000,50000))
    train_loader = DataLoader(train, batch_size=1)
    train_attacked_data = []
    train_labels = []
    train_attacked = []
    origin_data = []
    origin_label = []
    pgd = PGD(model=model, eps=radius/255, alpha=2/255, steps=10, random_start=True)
    i = 0
    k = 0
    # from torchattacks import PGD
    for images,labels in train_loader:
        k+=1
        images = images.to(device)
        labels = labels.to(device)
        # pgd = PGD(model=model, eps=radius/255, alpha=2/255, steps=10, random_start=True)
        # adv_images = pgd(images,labels)
        # with torch.no_grad():
        #     outputs = model(adv_images)
        # _, predicted = torch.max(outputs.data, 1)
        # if labels != predicted:
        #     print(f"attack success {i}")
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
    train_attack_data = train_attack_data[data_mask].reshape(-1,3,32,32)

    labels_mask = torch.ones_like(train_attack_labels,dtype=torch.bool)
    labels_mask[multi_number::multi_number+1] = False
    train_attack_labels = train_attack_labels[labels_mask]
    if general == True:
        torch.save((train_attack_data,train_attack_labels),f'./data/cifar10/test_attack_data_full_{net}_{radius}.pt')
        torch.save((origin_data,origin_label),f'./data/cifar10/origin_data_{net}_{radius}.pt')
        torch.save((train_repair_data,train_repair_labels),f'./data/cifar10/train_attack_data_full_{net}_{radius}.pt')
    else:
        torch.save((origin_data,origin_label),f'./data/cifar10/origin_data_{net}_{radius}.pt')
        torch.save((train_repair_data,train_repair_labels),f'./data/cifar10/train_attack_data_full_{net}_{radius}.pt')
        torch.save((train_attack_data,train_attack_labels),f'./data/cifar10/test_attack_data_full_{net}_{radius}.pt')
    with open(f'./data/cifar10/origin_data_{net}_{radius}.txt','a') as f:
        f.write(str(k))
        f.close()




def adv_training(net, radius, data_num, device,radius_bit = 8):
    # device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    if net == 'vgg19':
        model = VGG('VGG19').to(device)
        state = torch.load(f"/home/chizm/PatchART/model/cifar10/vgg19.pth")
        model.load_state_dict(state)
    elif net == 'resnet18':
        model = resnet18(num_classes=10).to(device)
        state = torch.load(f"/home/chizm/PatchART/model/cifar10/resnet18.pth")
        model.load_state_dict(state)



    train_attack_data,train_attack_labels = torch.load(f'./data/cifar10/train_attack_data_full_{net}_{radius_bit}.pt',map_location=device)
    train_attack_data = train_attack_data[:data_num]
    train_attack_labels = train_attack_labels[:data_num]
    # test_attack_data,test_attack_labels = torch.load(f'./data/cifar10/test_attack_data_full_{net}_{radius_bit}.pt',map_location=device)
    # test_attack_data = test_attack_data[:data_num]
    # test_attack_labels = test_attack_labels[:data_num]

    train_data, train_label = torch.load(f'./data/cifar10/train_norm.pt',map_location=device)
    train_dataset = torch.utils.data.TensorDataset(train_data,train_label)
    
    # dataset
    train_attack_dataset = torch.utils.data.TensorDataset(train_attack_data,train_attack_labels)
    # test_attack_dataset = torch.utils.data.TensorDataset(test_attack_data,test_attack_labels)
    # data loader
    train_loader = DataLoader(train_dataset, batch_size=128)
    train_attack_loader = DataLoader(train_attack_dataset, batch_size=50)
    # test_attack_loader = DataLoader(test_attack_dataset, batch_size=128)
    # train
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    model.train()
    for epoch in range(200):
        print(f"adv-training epoch {epoch}")
        # epoch_loss = 0
        # correct, total = 0,0
        # for inputs,labels in train_attack_loader:
        loss_sum = 0
        for (repair_input, repair_label), (origin_input, origin_label) in zip(train_attack_loader, train_loader):
            # inputs,labels = inputs.to(device),labels.to(device)
            repair_input = repair_input.to(device)
            repair_label = repair_label.to(device)
            origin_input = origin_input.to(device)
            origin_label = origin_label.to(device)
            optimizer.zero_grad()
            repair_output = model(repair_input)
            loss = criterion(repair_output, repair_label)
            loss.backward()
            optimizer.step()    
            loss_sum += loss.item()
            origin_output = model(origin_input)
            loss = criterion(origin_output, origin_label)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{200}], Loss: {loss_sum/len(train_attack_loader):.4f}')
        # for i in range(train_nbatch):
        #     inputs,labels = iter_train.__next__()
            # inputs = inputs.to(device)
            # labels = labels.to(device)
            # optimizer.zero_grad()
            # outputs = model(inputs)
            # loss = criterion(outputs, labels)
            # loss.backward()
            # optimizer.step()
        scheduler.step()
    
    return model.eval()


def grad_none(net, radius):
    # load
    origin_data,origin_label = torch.load(f'./data/cifar10/origin_data_{net}_{radius}.pt')
    train_attack_data,train_attack_labels = torch.load(f'./data/cifar10/train_attack_data_full_{net}_{radius}.pt')
    test_attack_data,test_attack_labels = torch.load(f'./data/cifar10/test_attack_data_full_{net}_{radius}.pt')
    # grad none
    origin_data.requires_grad = False
    origin_label.requires_grad = False
    train_attack_data.requires_grad = False
    train_attack_labels.requires_grad = False
    test_attack_data.requires_grad = False
    test_attack_labels.requires_grad = False
    # save
    torch.save((origin_data,origin_label),f'./data/cifar10/origin_data_{net}_{radius}.pt')
    torch.save((train_attack_data,train_attack_labels),f'./data/cifar10/train_attack_data_full_{net}_{radius}.pt')
    torch.save((test_attack_data,test_attack_labels),f'./data/cifar10/test_attack_data_full_{net}_{radius}.pt')





from torch import Tensor
def get_bitmap(in_lb: Tensor, in_ub: Tensor, in_bitmap: Tensor, batch_inputs: Tensor, device):
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
            is_in = (batch_inputs_clone >= in_lb) & (batch_inputs_clone <= in_ub)
            is_in = is_in.all(dim=-1) # every input is in the region of property, batch * n_prop
        elif len(in_lb.shape) == 4:
            if in_lb.shape[0] > 600:
                is_in_list = []
                for i in range(batch_inputs_clone.shape[0]):
                    batch_inputs_compare_datai = batch_inputs_clone[i].clone().expand(in_lb.shape[0], in_lb.shape[1], in_lb.shape[2], in_lb.shape[3])
                    is_in_datai = (batch_inputs_compare_datai >= in_lb) & (batch_inputs_compare_datai <= in_ub)
                    is_in_datai = is_in_datai.all(dim=(-1)).all(dim=(-1)).all(dim=(-1)) # every input is in the region of property, batch * n_prop
                    is_in_list.append(is_in_datai)
                is_in = torch.stack(is_in_list, dim=0)
            else:
                batch_inputs_clone = batch_inputs_clone.expand(batch_inputs.shape[0], in_lb.shape[0], in_lb.shape[1], in_lb.shape[2], in_lb.shape[3])
                is_in = (batch_inputs_clone >= in_lb) & (batch_inputs_clone <= in_ub)
                is_in = is_in.all(dim=(-1)).all(dim=(-1)).all(dim=(-1)) # every input is in the region of property, batch * n_prop
        # convert to bitmap
        bitmap = torch.zeros((batch_inputs.shape[0], in_bitmap.shape[1]),device=device).to(torch.uint8)
        # is in is a batch * in_bitmap.shape[0] tensor, in_bitmap.shape[1] is the number of properties
        # the every row of is_in is the bitmap of the input which row of in_bitmap is allowed
        bitmap_i, inbitmap_j =  is_in.nonzero(as_tuple=True)
        if bitmap_i.shape[0] != 0:
            bitmap[bitmap_i, :] = in_bitmap[inbitmap_j, :]
        else:
            pass

        return bitmap

# def compare_pgd_step_length(net, patch_format, 
#                             radius, repair_number):
#     '''
#     use the length of pgd steps to compare the hardness of attacking two model respectively
#     the model1 is origin model, model2 is repaired model
#     '''
#     # load net
#     from art.repair_moudle import Netsum
#     from DiffAbs.DiffAbs import deeppoly
#     from mnist.mnist_utils import MnistNet_CNN_small,MnistNet_FNN_small, MnistNet_FNN_big, Mnist_patch_model,MnistProp
#     device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
#     print(f'Using {device} device')
#     if net == 'vgg19':
#         model1 = VGG('VGG19').to(device)
#         state = torch.load(f"/home/chizm/PatchART/model/cifar10/vgg19.pth")
#         model1.load_state_dict(state)
#     elif net == 'resnet18':
#         model1 = ResNet18().to(device)
#         state = torch.load(f"/home/chizm/PatchART/model/cifar10/resnet18.pth")
#         model1.load_state_dict(state)

  



#     model1.to(device)
#     patch_lists = []
#     for i in range(repair_number):
#         if patch_format == 'small':
#             patch_net = Mnist_patch_model(dom=deeppoly, name = f'small patch network {i}')
#         elif patch_format == 'big':
#             patch_net = Mnist_patch_model(dom=deeppoly,name = f'big patch network {i}')
#         patch_net.to(device)
#         patch_lists.append(patch_net)
#     model2 =  Netsum(deeppoly, target_net = model1, patch_nets= patch_lists, device=device)
#     model2.load_state_dict(torch.load(f"/home/chizm/PatchART/model/patch_format/Mnist-{net}-repair_number{repair_number}-rapair_radius{radius}-{patch_format}.pt",map_location=device))

#     model3 = adv_training(net,radius, data_num=repair_number, device=device)


#     # load data
#     datas,labels = torch.load(f'/home/chizm/PatchART/data/MNIST/processed/origin_data_{net}_{radius}.pt',map_location=device)
#     # return
    
#     datas = datas[:repair_number]
#     labels = labels[:repair_number]

#     # pgd
#     pgd1 = PGD(model=model1, eps=radius, alpha=2/255, steps=50, random_start=True)
#     pgd2 = PGD(model=model2, eps=radius, alpha=2/255, steps=50, random_start=True)
#     pgd3 = PGD(model=model3, eps=radius, alpha=2/255, steps=50, random_start=True)

#     # attack
#     ori_step = 0
#     repair_step = 0
#     pgd_step = 0

#     # get bitmap
#     from art.prop import AndProp
#     from art.bisecter import Bisecter
#     repairlist = [(data[0],data[1]) for data in zip(datas, labels)]
#     repair_prop_list = MnistProp.all_props(deeppoly, DataList=repairlist, input_shape= datas.shape[1:], radius= radius)
#     # get the all props after join all l_0 ball feature property
#     # TODO squeeze the property list, which is the same as the number of label
#     all_props = AndProp(props=repair_prop_list)
#     # v = Bisecter(deeppoly, all_props)
#     in_lb, in_ub = all_props.lbub(device)
#     in_bitmap = all_props.bitmap(device)

#     p1 = 0
#     p2 = 0
#     p3 = 0

#     for image, label in zip(datas,labels):
#         image = image.unsqueeze(0).to(device)
#         label = label.unsqueeze(0).to(device)
        
#         step1, ori_acc = pgd1.forward_sumsteps(image,label)
#         step2, repair_acc = pgd2.forward_sumsteps(image,label, device=device, bitmap = [in_lb, in_ub, in_bitmap])
#         step3, adt_acc = pgd3.forward_sumsteps(image,label)
#         ori_step += step1
#         repair_step += step2
#         pgd_step += step3
#         if ori_acc == 1:
#             p1 += 1
#         if repair_acc == 1:
#             p2 += 1
#         if adt_acc == 1:
#             p3 += 1
            
    
#     print(f"ori_step {ori_step}, repair_step {repair_step}, pgd_step {pgd_step} \\ ori:{p1}, patch:{p2}, adv-train:{p3}")
#     with open(f'./data/MNIST/processed/compare_pgd_step_length.txt','a') as f:
#         f.write(f"For {net} {radius} {data} {patch_format}: \\ ori_step {ori_step}, repair_step {repair_step}, pgd_step {pgd_step} \\ ori:{p1}, patch:{p2}, adv-train:{p3} \\ \n")


from art.repair_moudle import Netsum, NetFeatureSumPatch
from DiffAbs.DiffAbs import deeppoly
    # from mnist.mnist_utils import MnistNet_CNN_small,MnistNet_FNN_small, MnistNet_FNN_big, Mnist_patch_model,MnistProp
from cifar10_utils import Cifar_feature_patch_model,CifarProp




def compare_autoattack(net, 
                            radius_bit, repair_number):
    '''
    use the length of pgd steps to compare the hardness of attacking two model respectively
    the model1 is origin model, model2 is repaired model
    '''
    # load net

    radius= radius_bit/255
    
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    print(f'Using {device} device')
    if net == 'vgg19':
        model1 = VGG('VGG19').to(device)
        state = torch.load(f"/home/chizm/PatchART/model/cifar10/vgg19.pth")
        model1.load_state_dict(state)
    elif net == 'resnet18':
        model1 = resnet18(num_classes=10).to(device)
        state = torch.load(f"/home/chizm/PatchART/model/cifar10/resnet18.pth")
        model1.load_state_dict(state)
    model1.to(device)
    model1.eval()


    if net == 'vgg19':
        frontier = VGG('VGG19').to(device)
        frontier_state = torch.load(f"/home/chizm/PatchART/model/cifar10/vgg19.pth")
        frontier.load_state_dict(frontier_state)
    elif net == 'resnet18':
        frontier = Resnet_model(dom= deeppoly).to(device)
        frontier_state = torch.load(f"/home/chizm/PatchART/model/cifar10/resnet18.pth")
        frontier.load_state_dict(frontier_state)
    frontier.to(device)

    frontier, rear  = frontier.split()

    patch_lists = []
    for i in range(repair_number):
        # if patch_format == 'small':
        patch_net = Cifar_feature_patch_model(dom=deeppoly, name = f'feature patch network {i}', input_dimension=512)
        # elif patch_format == 'big':
        #     patch_net = Cifar_feature_patch_model(dom=deeppoly,name = f'big patch network {i}')
        patch_net.to(device)
        patch_lists.append(patch_net)
    rear =  Netsum(deeppoly, target_net = rear, patch_nets= patch_lists, device=device)

    rear.load_state_dict(torch.load(f"/home/chizm/PatchART/model/cifar10_patch_format/Cifar-{net}-feature-repair_number{repair_number}-rapair_radius{radius_bit}.pt",map_location=device))
    model2 = NetFeatureSumPatch(feature_sumnet=rear, feature_extractor=frontier)
    torch.save(model2.state_dict(),f"/home/chizm/PatchART/model/cifar10_patch_format/Cifar-{net}-full-repair_number{repair_number}-rapair_radius{radius_bit}-feature_sumnet.pt")
    model2.eval()

    model3 = adv_training(net,radius, data_num=repair_number, device=device, radius_bit=radius_bit)


    # load data
    datas,labels = torch.load(f'/home/chizm/PatchART/data/cifar10/origin_data_{net}_{radius_bit}.pt',map_location=device)
    # return
    
    datas = datas[:repair_number]
    labels = labels[:repair_number]

    # pgd
    # pgd1 = PGD(model=model1, eps=radius, alpha=2/255, steps=50, random_start=True)
    # pgd2 = PGD(model=model2, eps=radius, alpha=2/255, steps=50, random_start=True)
    # pgd3 = PGD(model=model3, eps=radius, alpha=2/255, steps=50, random_start=True)
    from torchattacks import AutoAttack


    # attack
    # ori_step = 0
    # repair_step = 0
    # pgd_step = 0

    # get bitmap
    from art.prop import AndProp
    from art.bisecter import Bisecter
    repairlist = [(data[0],data[1]) for data in zip(datas, labels)]
    repair_prop_list = CifarProp.all_props(deeppoly, DataList=repairlist, input_shape= datas.shape[1:], radius= radius)
    # get the all props after join all l_0 ball feature property
    # TODO squeeze the property list, which is the same as the number of label
    all_props = AndProp(props=repair_prop_list)
    # v = Bisecter(deeppoly, all_props)
    in_lb, in_ub = all_props.lbub(device)
    in_bitmap = all_props.bitmap(device)

    bitmap = get_bitmap(in_lb, in_ub, in_bitmap, datas, device)

    p1 = 0
    p2 = 0
    p3 = 0

    for ith, (image, label) in enumerate(zip(datas,labels)):
        image = image.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)

        at1 = AutoAttack(model1, norm='Linf', eps=radius, version='standard', verbose=False)
        adv_images1 = at1(image, label)
        if model1(adv_images1).argmax(dim=1)!= label:
            print("success1")
            p1 += 1
        else:
            print("fail")
        at2 = AutoAttack(model2, norm='Linf', eps=radius, version='standard', verbose=False, bitmap=bitmap)
        adv_images2 = at2(image, label)
        if model2(adv_images2, bitmap[ith]).argmax(dim=1) != label:
            print("success2")
            p2 += 1
        else:
            print("fail")
        at3 = AutoAttack(model3, norm='Linf', eps=radius, version='standard', verbose=False)
        adv_images3 = at3(image, label)
        if model3(adv_images3).argmax(dim=1) != label:
            print("success3")
            p3 += 1
        else:
            print("fail")
        
        # step1, ori_acc = pgd1.forward_sumsteps(image,label)
        # step2, repair_acc = pgd2.forward_sumsteps(image,label, device=device, bitmap = [in_lb, in_ub, in_bitmap])
        # step3, adt_acc = pgd3.forward_sumsteps(image,label)
        # ori_step += step1
        # repair_step += step2
        # pgd_step += step3
        # if ori_acc == 1:
        #     p1 += 1
        # if repair_acc == 1:
        #     p2 += 1
        # if adt_acc == 1:
        #     p3 += 1
            
    
    # print(f"ori_step {ori_step}, repair_step {repair_step}, pgd_step {pgd_step} \\ ori:{p1}, patch:{p2}, adv-train:{p3}")
    with open(f'/home/chizm/PatchART/results/cifar10/repair/autoattack/compare_autoattack_ac.txt','a') as f:
        f.write(f"For {net} {repair_number} {radius} : \\  ori:{p1}, patch:{p2}, adv-train:{p3} \\ \n")

def patch_label_autoattack(net, 
                            radius_bit, repair_number,device):
    '''
    use the length of pgd steps to compare the hardness of attacking two model respectively
    the model1 is origin model, model2 is repaired model
    '''
    # load net

    radius= radius_bit/255
    
    device = device if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    # if net == 'vgg19':
    #     model1 = VGG('VGG19').to(device)
    #     state = torch.load(f"/home/chizm/PatchART/model/cifar10/vgg19.pth")
    #     model1.load_state_dict(state)
    # elif net == 'resnet18':
    #     model1 = resnet18(num_classes=10).to(device)
    #     state = torch.load(f"/home/chizm/PatchART/model/cifar10/resnet18.pth")
    #     model1.load_state_dict(state)
    # model1.to(device)
    # model1.eval()


    if net == 'vgg19':
        frontier = VGG('VGG19').to(device)
        frontier_state = torch.load(f"/home/chizm/PatchART/model/cifar10/vgg19.pth")
        frontier.load_state_dict(frontier_state)
    elif net == 'resnet18':
        frontier = Resnet_model(dom= deeppoly).to(device)
        frontier_state = torch.load(f"/home/chizm/PatchART/model/cifar10/resnet18.pth")
        frontier.load_state_dict(frontier_state)
    frontier.to(device)

    frontier, rear  = frontier.split()

    patch_lists = []
    for i in range(10):
        # if patch_format == 'small':
        patch_net = Cifar_feature_patch_model(dom=deeppoly, name = f'feature patch network {i}', input_dimension=512)
        # elif patch_format == 'big':
        #     patch_net = Cifar_feature_patch_model(dom=deeppoly,name = f'big patch network {i}')
        patch_net.to(device)
        patch_lists.append(patch_net)
    rear =  Netsum(deeppoly, target_net = rear, patch_nets= patch_lists, device=device)

    rear.load_state_dict(torch.load(f"/home/chizm/PatchART/model/cifar10_label_format/Cifar-{net}-feature-repair_number{repair_number}-rapair_radius{radius_bit}.pt",map_location=device))
    model2 = NetFeatureSumPatch(feature_sumnet=rear, feature_extractor=frontier)
    torch.save(model2.state_dict(),f"/home/chizm/PatchART/model/cifar10_label_format/Cifar-{net}-full-repair_number{repair_number}-rapair_radius{radius_bit}-feature_sumnet.pt")
    model2.eval()

    # model3 = adv_training(net,radius, data_num=repair_number, device=device, radius_bit=radius_bit)


    # load data
    datas,labels = torch.load(f'/home/chizm/PatchART/data/cifar10/origin_data_{net}_{radius_bit}.pt',map_location=device)
    # return
    
    datas = datas[:repair_number]
    labels = labels[:repair_number]

    # pgd
    # pgd1 = PGD(model=model1, eps=radius, alpha=2/255, steps=50, random_start=True)
    # pgd2 = PGD(model=model2, eps=radius, alpha=2/255, steps=50, random_start=True)
    # pgd3 = PGD(model=model3, eps=radius, alpha=2/255, steps=50, random_start=True)
    from torchattacks import AutoAttack


    # attack
    # ori_step = 0
    # repair_step = 0
    # pgd_step = 0

    # get bitmap
    from art.prop import AndProp
    from art.bisecter import Bisecter
    repairlist = [(data[0],data[1]) for data in zip(datas, labels)]
    repair_prop_list = CifarProp.all_props(deeppoly, DataList=repairlist, input_shape= datas.shape[1:], radius= radius)
    # get the all props after join all l_0 ball feature property
    # TODO squeeze the property list, which is the same as the number of label
    all_props = AndProp(props=repair_prop_list)
    # v = Bisecter(deeppoly, all_props)
    in_lb, in_ub = all_props.lbub(device)
    in_bitmap = all_props.bitmap(device)

    bitmap = get_bitmap(in_lb, in_ub, in_bitmap, datas, device)

    p1 = 0
    p2 = 0
    p3 = 0

    for ith, (image, label) in enumerate(zip(datas,labels)):
        image = image.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)

        # at1 = AutoAttack(model1, norm='Linf', eps=radius, version='standard', verbose=False)
        # adv_images1 = at1(image, label)
        # if model1(adv_images1).argmax(dim=1)!= label:
        #     print("success1")
        #     p1 += 1
        # else:
        #     print("fail")
        at2 = AutoAttack(model2, norm='Linf', eps=radius, version='standard', verbose=False, bitmap=bitmap)
        adv_images2 = at2(image, label)
        if model2(adv_images2, bitmap[ith]).argmax(dim=1) != label:
            print("success2")
            p2 += 1
        else:
            print("fail")
        # at3 = AutoAttack(model3, norm='Linf', eps=radius, version='standard', verbose=False)
        # adv_images3 = at3(image, label)
        # if model3(adv_images3).argmax(dim=1) != label:
        #     print("success3")
        #     p3 += 1
        # else:
        #     print("fail")
        
        # step1, ori_acc = pgd1.forward_sumsteps(image,label)
        # step2, repair_acc = pgd2.forward_sumsteps(image,label, device=device, bitmap = [in_lb, in_ub, in_bitmap])
        # step3, adt_acc = pgd3.forward_sumsteps(image,label)
        # ori_step += step1
        # repair_step += step2
        # pgd_step += step3
        # if ori_acc == 1:
        #     p1 += 1
        # if repair_acc == 1:
        #     p2 += 1
        # if adt_acc == 1:
        #     p3 += 1
            
    
    # print(f"ori_step {ori_step}, repair_step {repair_step}, pgd_step {pgd_step} \\ ori:{p1}, patch:{p2}, adv-train:{p3}")
    with open(f'/home/chizm/PatchART/results/cifar10/repair/autoattack/compare_autoattack_ac.txt','a') as f:
        f.write(f"For {net} {radius} {repair_number} : label:{p2}\n")
if __name__ == '__main__':
    # training('vgg19','cuda:0')
    # get_dataset()

    # for net in ['resnet18']:
    #     # trasfer_state_dict(net)
        for radius in [4,8]:
            for net in ['vgg19','resnet18']:
                # trasfer_state_dict(net)
        #         for repair_num in [500,1000]:
                    # compare_autoattack(net,radius_bit=radius, repair_number=repair_num)
        # for radius in [8]:
        #     for net in ['vgg19']:
        #         for repair_num in [50,100,200,500,1000]:
                # for repair_num in [50,100,200,500,1000]:
                # for dataset_type in ['repair', 'attack_test']:
                #     test_dataset(net,radius=radius, dataset_type=dataset_type,device='cuda:1')
    #         if net == 'vgg19' and radius == 8:
    #             continue
                    # pgd_get_data(net,radius=radius,multi_number=10,data_num=1000,general=False)
                    # grad_none(net,radius=radius)
        # for net in ['resnet18']:
        #     for radius in [4,8]:
                for repair_num in [50,100,200,500,1000]:  
                    # compare_autoattack(net,radius_bit=radius, repair_number=repair_num)
                    patch_label_autoattack(net,radius_bit=radius, repair_number=repair_num,device='cuda:1')
        # for net in ['resnet18']:
        #     for radius in [4]:
        #         for repair_num in [50,100,200,500,1000]:
    # pgd_get_data('resnet18',radius=8,multi_number=10,data_num=1000,general=False)
    # grad_none('vgg19',radius=8)
    # adv_training('vgg19',radius=8,data_num=1000,device='cuda:0')
    # from autoattack import AutoAttack