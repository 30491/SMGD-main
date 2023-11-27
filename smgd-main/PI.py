### vt方法可以适用于densenet和resnet网络结构，因为vt是对图片输入进行处理的，并不特定于某个网络结构

import torch
import argparse
import os
import numpy as np
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from utils_data import SubsetImageNet, save_images
from utils_iaa import resnet152
import torch.nn.functional as F
import pretrainedmodels
import random
model_names = sorted(name for name in pretrainedmodels.__dict__
                     if not name.startswith("__")
                     and name.islower()
                     and callable(pretrainedmodels.__dict__[name]))

class Preprocessing_Layer(torch.nn.Module):
    def __init__(self, mean, std):
        super(Preprocessing_Layer, self).__init__()
        self.mean = mean
        self.std = std

    def preprocess(self, img, mean, std):
        img = img.clone()
        #img /= 255.0

        img[:,0,:,:] = (img[:,0,:,:] - mean[0]) / std[0]
        img[:,1,:,:] = (img[:,1,:,:] - mean[1]) / std[1]
        img[:,2,:,:] = (img[:,2,:,:] - mean[2]) / std[2]

        #img = img.transpose(1, 3).transpose(2, 3)
        return(img)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        res = self.preprocess(x, self.mean, self.std)
        return res

# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]

class Ensemble(nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = models
        self.normalize = len(models)

    def forward(self, x):
        output = 0
        for i in range(len(self.models)):
            output += F.softmax(self.models[i](x), dim=1)
        output /= self.normalize
        return torch.log(output)

def load_pretrained_model(model, pretrained_dict):
	model_dict = model.state_dict()
	# 1. filter out unnecessary keys
	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
	# 2. overwrite entries in the existing state dict
	model_dict.update(pretrained_dict)
	# 3. load the new state dict
	model.load_state_dict(model_dict)


parser = argparse.ArgumentParser(description='PyTorch ImageNet Attack Evaluation')#

parser.add_argument('--input_dir', default='./dataset/dataset/SubImageNetVal', help='the path of original dataset')
parser.add_argument('--output_dir', default='./output/densenet201/BIM/pi_t_', help='the path of the saved dataset')
parser.add_argument('--arch', default='densenet201',help='source model for black-box attack evaluation',choices=model_names)

parser.add_argument('--ensemble', default=1, type=int) # 1为True
parser.add_argument('--ensemble_num', default='8', type=int) ### 集成网络的个数，到底集成几个学生网络 每次实验前都检查学生网路权重
parser.add_argument('--snet_dir1', default='./result/densenet201/20/1gpu_checkpoint.pth.tar', help='the path of snet1') ## batchsize = 30
parser.add_argument('--snet_dir2', default='./result/densenet201/22/1gpu_checkpoint.pth.tar', help='the path of snet2')
parser.add_argument('--snet_dir3', default='./result/densenet201/24/1gpu_checkpoint.pth.tar', help='the path of snet3')
parser.add_argument('--snet_dir4', default='./result/densenet201/26/1gpu_checkpoint.pth.tar', help='the path of snet4')
parser.add_argument('--snet_dir5', default='./result/densenet201/28/1gpu_checkpoint.pth.tar', help='the path of snet5') ## batchsize = 9
parser.add_argument('--snet_dir6', default='./result/densenet201/30/1gpu_checkpoint.pth.tar', help='the path of snet6')
parser.add_argument('--snet_dir7', default='./result/densenet201/32/1gpu_checkpoint.pth.tar', help='the path of snet7') ## batchsize = 6
parser.add_argument('--snet_dir8', default='./result/densenet201/18/1gpu_checkpoint.pth.tar', help='the path of snet8')
parser.add_argument('--cuda', type=int, default=[1,2,3])
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
args = parser.parse_args()

### load dataset
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])
data_set = SubsetImageNet(root='./dataset/dataset/SubImageNetVal', transform=transform_test)
data_loader = torch.utils.data.DataLoader(data_set, batch_size=10, shuffle=False, **kwargs)

### 图片保存的地方
ensemble_num = args.ensemble_num
print('ensemble',args.ensemble)
print('ensemble_num',ensemble_num)
output_dir = args.output_dir + str(ensemble_num)
print('output_dir',output_dir)
if os.path.exists(str(output_dir)) == False:
    os.makedirs(str(output_dir))

### parameter setting
check_point = 5
num_iteration = 10
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
preprocess_layer = Preprocessing_Layer(mean,std)
pos = np.zeros(num_iteration // check_point)
epsilon = 16.0 / 255.0
# step_size = 2.0 / 255 # PGD
step_size = 1.6 / 255 # BIM
multi_copies = 5 # vt的参数设置，参考为5

#### load model
#### 加载模型，从本地加载

# 加载教师网络
tnet = models.__dict__[args.arch](pretrained= True)
tmodel = nn.Sequential(Normalize(mean=mean, std=std), tnet)
tmodel.cuda()
tmodel = torch.nn.DataParallel(tmodel,device_ids=[0,1,2])
tmodel.eval()

# 加载学生网络
if args.ensemble == True:
    ####################################
    # net2 = pretrainedmodels.__dict__[args.arch](num_classes=1000, pretrained='imagenet')
    ### student model
    # ### 加载第一个学生网络
    snet1 = models.__dict__[args.arch](pretrained=False)
    smodel1 = nn.Sequential(Normalize(mean=mean, std=std), snet1)
    checkpoint1 = torch.load(args.snet_dir1)  # initial parameters of student model
    load_pretrained_model(smodel1, checkpoint1['snet'])
    smodel1 = torch.nn.DataParallel(smodel1, list(range(3))).cuda()
    # smodel1.cuda()
    smodel1.eval()
    ## snet2
    snet2 = models.__dict__[args.arch](pretrained=False)
    smodel2 = nn.Sequential(Normalize(mean=mean, std=std), snet2)
    checkpoint2 = torch.load(args.snet_dir2)  # initial parameters of student model
    load_pretrained_model(smodel2, checkpoint2['snet'])
    smodel2 = torch.nn.DataParallel(smodel2, list(range(3))).cuda()
    # smodel2.cuda()
    smodel2.eval()
    # ### snet3
    snet3 = models.__dict__[args.arch](pretrained=False)
    smodel3 = nn.Sequential(Normalize(mean=mean, std=std), snet3)
    checkpoint3 = torch.load(args.snet_dir3)  # initial parameters of student model
    load_pretrained_model(smodel3, checkpoint3['snet'])
    smodel3 = torch.nn.DataParallel(smodel3, list(range(3))).cuda()
    smodel3.eval()
    # ### snet4
    snet4 = models.__dict__[args.arch](pretrained=False)
    smodel4 = nn.Sequential(Normalize(mean=mean, std=std), snet4)
    checkpoint4 = torch.load(args.snet_dir4)  # initial parameters of student model
    load_pretrained_model(smodel4, checkpoint4['snet'])
    smodel4 = torch.nn.DataParallel(smodel4, list(range(3))).cuda()
    smodel4.eval()
    # ### snet5
    snet5 = models.__dict__[args.arch](pretrained=False)
    smodel5 = nn.Sequential(Normalize(mean=mean, std=std), snet5)
    checkpoint5 = torch.load(args.snet_dir5)  # initial parameters of student model
    load_pretrained_model(smodel5, checkpoint5['snet'])
    smodel5 = torch.nn.DataParallel(smodel5, list(range(3))).cuda()
    smodel5.eval()
    ### snet6
    snet6 = models.__dict__[args.arch](pretrained=False)
    smodel6 = nn.Sequential(Normalize(mean=mean, std=std), snet6)
    checkpoint6 = torch.load(args.snet_dir6)  # initial parameters of student model
    load_pretrained_model(smodel6, checkpoint6['snet'])
    smodel6 = torch.nn.DataParallel(smodel6, list(range(3))).cuda()
    smodel6.eval()
    # # # ### snet7
    snet7 = models.__dict__[args.arch](pretrained=False)
    smodel7 = nn.Sequential(Normalize(mean=mean, std=std), snet7)
    checkpoint7 = torch.load(args.snet_dir7)  # initial parameters of student model
    load_pretrained_model(smodel7, checkpoint7['snet'])
    smodel7 = torch.nn.DataParallel(smodel7, list(range(3))).cuda()
    smodel7.eval()
    # # ### snet8
    snet8 = models.__dict__[args.arch](pretrained=False)
    smodel8 = nn.Sequential(Normalize(mean=mean, std=std), snet8)
    checkpoint8 = torch.load(args.snet_dir8)  # initial parameters of student model
    load_pretrained_model(smodel8, checkpoint8['snet'])
    smodel8 = torch.nn.DataParallel(smodel8, list(range(3))).cuda()
    smodel8.eval()

    #################################### 集成网络

    print('ensemble_num',ensemble_num)
    # print(type(ensemble_num))
    # exit()
    if ensemble_num == 1:
        print('ensemble_num = ', ensemble_num)
        ensemble = Ensemble([tmodel, smodel1]).to(device)
    if ensemble_num == 2:
        print('ensemble_num = ', ensemble_num)
        ensemble = Ensemble([tmodel, smodel1, smodel2]).to(device)
    if ensemble_num == 3:
        print('ensemble_num = ', ensemble_num)
        ensemble = Ensemble([tmodel, smodel1, smodel2, smodel3]).to(device)
    if ensemble_num == 4:
        print('ensemble_num = ', ensemble_num)
        ensemble = Ensemble([tmodel, smodel1, smodel2, smodel3, smodel4]).to(device)
    if ensemble_num == 5:
        print('ensemble_num = ', ensemble_num)
        ensemble = Ensemble([tmodel, smodel1, smodel2, smodel3, smodel4, smodel5]).to(device)
    if ensemble_num == 6:
        print('ensemble_num = ', ensemble_num)
        ensemble = Ensemble([tmodel, smodel1, smodel2, smodel3, smodel4, smodel5, smodel6]).to(device)
    if ensemble_num == 7:
        print('ensemble_num = ', ensemble_num)
        ensemble = Ensemble([tmodel, smodel1, smodel2, smodel3, smodel4, smodel5, smodel6, smodel7]).to(device)
    if ensemble_num == 8:
        print('ensemble_num = ', ensemble_num)
        ensemble = Ensemble([tmodel, smodel1, smodel2, smodel3, smodel4, smodel5, smodel6, smodel7, smodel8]).to(device)

    print("--ensemble success--")


if args.ensemble == True:
    model  = ensemble
else:
    model = tmodel


for i, (images, labels ,idx) in enumerate(data_loader):
    images = images.to(device)
    labels = labels.to(device)
    img = images.clone()
    grad_pre = 0
    grad_t = 0
    for j in range(num_iteration):
        img_x = img + step_size * 1 * grad_t
        img_x.requires_grad_(True)
        logits = model(img_x)
        loss = nn.CrossEntropyLoss(reduction='sum')(logits, labels)
        loss.backward()
        grad_t = img_x.grad.clone()
        grad_t = grad_t / torch.mean(torch.abs(grad_t), (1, 2, 3), keepdim=True)
        input_grad = grad_t + 1 * grad_pre  # MI
        grad_pre = input_grad

        img_x.grad.zero_()
        img = img.data + step_size * torch.sign(input_grad)
        img = torch.where(img > images + epsilon, images + epsilon, img)
        img = torch.where(img < images - epsilon, images - epsilon, img)
        img = torch.clamp(img, min=0, max=1)

        flag = (j+1) % check_point
        if flag == 0:
            point = j // check_point
            pos[point] = pos[point] + sum(torch.argmax(model(img),dim=1) != labels).cpu().numpy()
        if j == 9:
            save_images(img.detach().cpu().numpy(), img_list=data_set.img_path, idx=idx, output_dir=output_dir)
print(pos)