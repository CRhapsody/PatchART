import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from vgg import VGG


# Define the Vgg19 model
def training(model_type:str,device):
    '''
    model: str, the name of the model
    '''
    if model_type == 'vgg19':
        model = VGG('VGG19')
        # model.classifier[6] = nn.Linear(4096, 10)
        model.to(device)


# split the last fc layer from vgg19
# model_last_fc = nn.Sequential(*list(vgg19_model.children())[-1:])





    # Load the CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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


def get_dataset():
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
    torch.save((trainset_inputs_split, trainset_labels_split),'/home/chizm/PatchART/data/cifar10/train_norm00.pt')
    # torch.save((trainset_inputs,trainset_labels),'/home/chizm/PatchART/data/MNIST/processed/train_norm00_full.pt')
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
    torch.save((testset_inputs,testset_labels),'/home/chizm/PatchART/data/cifar10/test_norm00.pt')




if __name__ == '__main__':
    training('vgg19','cuda:0')
    # get_dataset()