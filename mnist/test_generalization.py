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

from art.repair_moudle import Netsum
from DiffAbs.DiffAbs import AbsDom, AbsEle
from torch import Tensor, nn
from mnist.mnist_utils import MnistNet, Mnist_patch_model,MnistProp
from art.repair_moudle import Netsum
from DiffAbs.DiffAbs import deeppoly
from art.prop import AndProp
from art.bisecter import Bisecter

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

class NetEnsembleSum(Netsum):

    def forward(self, x, in_bitmap, out=None):
        # TODO not for batch input
        if out == None:
            out = self.target_net(x)
        else:
            assert isinstance(out, AbsEle), 'out should be AbsEle'
        
        if in_bitmap.any():
            for i,patch in enumerate(self.patch_nets):
                bits = in_bitmap[..., i]
                if not bits.any():
                # no one here needs to obey this property
                    continue

                ''' The default nonzero(as_tuple=True) returns a tuple, make scatter_() unhappy.
                    Here we just extract the real data from it to make it the same as old nonzero().squeeze(dim=-1).
                '''
                bits = bits.nonzero(as_tuple=True)[0]
                if isinstance(out, Tensor):
                    out[bits] += patch(x[bits])
                elif isinstance(out, AbsEle):
                    replace_item = out[bits] + patch(x[bits]) # may not only one prop
                    out.replace(in_bitmap[..., i], replace_item)
            return out.argmax(dim=1, keepdim=True)
        else:
            preds_list = []
            for i,patch in enumerate(self.patch_nets):
                # ensemble and statics all the output which patch nets add to the target net, then choose the class which has max votes
                assert isinstance(out, Tensor), 'out should be Tensor'
                out_for_ensemble = out + patch(x)
                preds = out_for_ensemble.argmax(dim=1, keepdim=True)
                preds_list.append(preds)
            preds = torch.cat(preds_list, dim=1)
            preds = torch.mode(preds, dim=1)[0].squeeze()
            return preds

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

def test_generalization(radius, repair_number):
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    model1 = NeuralNet().to(device)
    model1.load_state_dict(torch.load("/home/chizm/PatchART/model/mnist/mnist.pth"))

     
    net = MnistNet(dom=deeppoly)
    net.to(device)
    patch_lists = []
    for i in range(repair_number):
        patch_net = Mnist_patch_model(dom=deeppoly,
            name = f'patch network {i}')
        patch_net.to(device)
        patch_lists.append(patch_net)
    model2 =  NetEnsembleSum(deeppoly, target_net = net, patch_nets= patch_lists, device=device)
    model2.load_state_dict(torch.load(f"/home/chizm/PatchART/model/patch_format/Mnist-repair_number{repair_number}-rapair_radius{radius}-.pt",map_location=device))

    # model3 = adv_training(radius, data_num=repair_number)
    datas,labels = torch.load(f'/home/chizm/PatchART/data/MNIST/processed/origin_data_{radius}_{repair_number}.pt',map_location=device)
    test_data, test_labels = torch.load(f'/home/chizm/PatchART/data/MNIST/processed/train_attack_data_full_{radius}_{repair_number}_others.pt',map_location=device)

    repairlist = [(data[0],data[1]) for data in zip(datas, labels)]
    repair_prop_list = MnistProp.all_props(deeppoly, DataList=repairlist, input_shape= datas.shape[1:], radius= radius)
    # get the all props after join all l_0 ball feature property
    # TODO squeeze the property list, which is the same as the number of label
    all_props = AndProp(props=repair_prop_list)
    v = Bisecter(deeppoly, all_props)
    in_lb, in_ub = all_props.lbub(device)
    in_bitmap = all_props.bitmap(device)

    pred1_acc = 0
    pred2_acc = 0
    for image, label in zip(test_data,test_labels):
        image = image.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)
        # image = image.to(device)
        # label = label.to(device)
        bitmap = get_bitmap(in_lb, in_ub, in_bitmap, image)
        pred1 = model1(image).argmax(dim=1)
        pred2 = model2(image, bitmap)
        # pred3 = model3(image).argmax(dim=1)
        if pred1 != label:
            print('origin model error')
        else:
            pred1_acc += 1
        if pred2 != label:
            print('patch model error')
        else:
            pred2_acc += 1
        # if pred3 != label:
        #     print('adv model error')
        # else:
        #     pred3_acc += 1
    with open(f'/home/chizm/PatchART/data/MNIST/processed/ensemble_model_acc_{radius}_{repair_number}.txt','a') as f:
        f.write(f'origin model accuracy: {pred1_acc/len(test_data)}\n')
        f.write(f'patch model accuracy: {pred2_acc/len(test_data)}\n')
    # print(f'adv model accuracy: {pred3_acc/len(test_data)}')


def image_matching_with_ORB(test_image, radius, net):
    import cv2
    datas,labels = torch.load(f'/home/chizm/PatchART/data/MNIST/processed/origin_data_{net}_{radius}.pt',map_location=device)
    # the datas is a dataset, and we want to match the test_image with the datas to choose a best match
    # test_image is a tensor, and datas is a dataset
    test_image = test_image.cpu().numpy()
    datas = datas.cpu().numpy()
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(test_image,None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    max_match = 0
    max_index = 0
    for i,data in enumerate(datas):
        kp2, des2 = orb.detectAndCompute(data,None)
        # Match descriptors.
        matches = bf.match(des1,des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        if len(matches) > max_match:
            max_match = len(matches)
            max_index = i
    return max_index

if __name__ == '__main__':
    test_generalization(radius=0.1, repair_number=200)

