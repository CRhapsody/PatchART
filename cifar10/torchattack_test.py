import torch
import torchvision
import torchvision.transforms as transforms
from torchattacks import AutoAttack
from vgg import VGG
# Load the model
model = VGG('VGG19')
model.load_state_dict(torch.load('/home/chizm/PatchART/model/cifar10/vgg19.pth'))
model.to('cuda')
# Load the dataset
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)


testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)


adv=AutoAttack(model, norm='Linf', eps=2/255, version='standard', verbose=True)