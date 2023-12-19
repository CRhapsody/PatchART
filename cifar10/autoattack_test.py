import torch
import torchvision
import torchvision.transforms as transforms
from autoattack import AutoAttack
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

adversary = AutoAttack(model, norm='Linf', eps=2/255, version='standard', verbose=True)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)


testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

# Run AutoAttack
adversary.attacks_to_run = ['apgd-ce']
adversary.apgd.n_restarts = 0
adversary.apgd.n_iter = 200
# adversary.apgd.eps_iter = 2/255
# adversary.apgd.rand_init = True
# adversary.apgd.norm = np.inf
adversary.apgd.eps = 2/255
adversary.apgd.loss = 'ce'
adversary.apgd.verbose = True
# adversary.apgd.check_adv = False
# adversary.apgd.check_grad = False
# adversary.apgd.check_loss = False
for batch_idx, (inputs, targets) in enumerate(train_loader):
    step = 0
    if batch_idx >= 100:
        break
    adv_list = []




    while True:
        if step > 200:
            break


        adv = adversary.run_standard_evaluation_individual(inputs, targets, bs=1)
        if adv is None:
            continue
        adv_list.append(adv)

        if len(adv_list) >= 11:
            adv_cat = torch.cat(adv_list, dim=0)
            adv_cat_unique = torch.unique(adv_cat, dim=0)
            if len(adv_cat_unique) == 11:
                break
            else:
                continue
# train_repair_data = train_attack_data[multi_number::multi_number+1]
# train_repair_labels = train_attack_labels[multi_number::multi_number+1]
# data_mask = torch.ones_like(train_attack_data,dtype=torch.bool)
# data_mask[multi_number::multi_number+1] = False
# train_attack_data = train_attack_data[data_mask].reshape(-1,3,32,32)

# labels_mask = torch.ones_like(train_attack_labels,dtype=torch.bool)
# labels_mask[multi_number::multi_number+1] = False
# train_attack_labels = train_attack_labels[labels_mask]    


    print(f'success for {batch_idx}th image')
    print(adv)