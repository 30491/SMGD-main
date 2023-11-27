# encoding:utf-8
"""Implementation of sample attack."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from utils_data import SubsetImageNet, save_images
    # save_image_tensor2cv2
from torchvision import transforms as T
import torch.nn.functional as F
from torch.autograd import Variable as V
# from torch.autograd.gradcheck import zero_gradients
from torch.utils import data
import os
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import csv
from torch_nets import (
    tf_adv_inception_v3,
    tf_ens3_adv_inc_v3,
    tf_ens4_adv_inc_v3,
    tf_ens_adv_inc_res_v2,
    )


list_nets = [
    'tf_adv_inception_v3',
    'tf_ens3_adv_inc_v3',
    'tf_ens4_adv_inc_v3',
    'tf_ens_adv_inc_res_v2'
    ]

parser = argparse.ArgumentParser()

# parser.add_argument('--gpu', type=str, default='0', help='The ID of GPU to use.')
parser.add_argument('--input_dir', type=str, default='./output/resnet152/PGD/t_1', help='Input images.')
parser.add_argument('--model_dir', type=str, default='./torch_nets_weight/', help='Model weight directory.')
parser.add_argument('--csv', type=int, default=0, help='')

parser.add_argument("--batch_size", type=int, default=20, help="How many images process at one time.")
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

opt = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
use_cuda = not opt.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

def seed_torch(seed):
    """Set a random seed to ensure that the results are reproducible"""  
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

class Normalize(nn.Module):

    def __init__(self, mean=0, std=1, mode='tensorflow'):
        """
        mode:
            'tensorflow':convert data from [0,1] to [-1,1]
            'torch':(input - mean) / std
        """
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std
        self.mode = mode

    def forward(self, input):
        size = input.size()
        x = input.clone()

        if self.mode == 'tensorflow':
            x = x * 2.0 - 1.0  # convert data from [0,1] to [-1,1]
        elif self.mode == 'torch':
            for i in range(size[1]):
                x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
        return x

def get_model(net_name, model_dir):
    """Load converted model"""
    model_path = os.path.join(model_dir, net_name + '.npy')

    if net_name == 'tf_adv_inception_v3':
        net = tf_adv_inception_v3
    elif net_name == 'tf_ens3_adv_inc_v3':
        net = tf_ens3_adv_inc_v3
    elif net_name == 'tf_ens4_adv_inc_v3':
        net = tf_ens4_adv_inc_v3
    elif net_name == 'tf_ens_adv_inc_res_v2':
        net = tf_ens_adv_inc_res_v2
    else:
        print('Wrong model name!')

    model = nn.Sequential(
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        Normalize('tensorflow'),
        # Normalize(mean=mean,std=std,mode='torch'),
        net.KitModel(model_path).eval().cuda(),)
    return model


def get_models(list_nets, model_dir):
    """load models with dict"""
    nets = {}
    for net in list_nets:
        nets[net] = get_model(net, model_dir)
    return nets

def main():

    transform_test = transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
    ])
    inputs = SubsetImageNet(root=opt.input_dir, transform=transform_test)  # 数据集处理操作
    data_loader = torch.utils.data.DataLoader(inputs, batch_size=opt.batch_size, shuffle=False, **kwargs)  # 批处理操作
    input_num = len(inputs)
    # inputs = ImageNet(opt.input_dir, opt.input_csv, transforms)
    # data_loader = DataLoader(inputs, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)

    # Create models
    models = get_models(list_nets, opt.model_dir)
    # Initialization parameters
    correct_num = {}
    logits = {}
    for net in list_nets:
        correct_num[net] = 0

    # Start iteration
    # for images, label, idx in tqdm(data_loader):
    for batch_idx,(images, label, idx) in enumerate(data_loader):
    # for images, filename, label in tqdm(data_loader):
        label = label.cuda()
        images = images.cuda()

        # Prediction
        with torch.no_grad():
            for net in list_nets:
                logits[net] = models[net](images)
                correct_num[net] += (torch.argmax(logits[net][0], axis=1) == (label+1)).detach().sum().cpu()

    # Print attack success rate
    for net in list_nets:
        # print(input_num)
        # print(correct_num[net])
        # print('{} correct success rate: {:.2%}'.format(net, correct_num[net].float()/input_num))
        print('{} attack success rate: {:.2%}'.format(net, (input_num-correct_num[net].float())/input_num))
        print('============================================')

        # with open(os.path.join(opt.input_dir,'success.csv'), mode='a+', encoding="utf-8-sig", newline="") as f:
        #     # 基于打开的文件，创建 csv.DictWriter 实例，将 header 列表作为参数传入。
        #     csv_write = csv.writer(f)
        #     #  3.构建列表头
        #     csv_write.writerow([net])
        #     #  4.写入csv文件
        #     csv_write.writerow([(input_num-correct_num[net].float())/input_num*100])
        if opt.csv==True:
            data = pd.read_csv(os.path.join('./output/densenent201/BIM/', 'success.csv'), float_precision="round_trip")
            data1 = [round(float((input_num-correct_num[net].float())/input_num*100),2)]
            data[net] = data1
            data.to_csv(os.path.join('./output/densenent201/BIM/', 'success.csv'), mode='w', index=False)


if __name__ == '__main__':
    seed_torch(0)
    main()
