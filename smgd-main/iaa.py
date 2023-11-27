"""目前只有resnet系列的超参数，densenet系列的没有，并且实验中模型不能从torchvision库中加载，
    只能在本地加载，因为模型需要修改
    我们的方法和它们结合，可以使用resnet152+smgd1
    因为smgd8内存占用太大 """

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

save_dir = os.path.join('./output/resnet152/PGD/iaa_t1')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Attack Evaluation')#

parser.add_argument('--input_dir', default='./dataset/dataset/SubImageNetVal', help='the path of original dataset')
parser.add_argument('--cuda', type=int, default=[1,2,3])
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
args = parser.parse_args()

# 数据集加载
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])
data_set = SubsetImageNet(root='./dataset/dataset/SubImageNetVal', transform=transform_test)
data_loader = torch.utils.data.DataLoader(data_set, batch_size=50, shuffle=False, **kwargs)

### 超参数设置
check_point = 5
num_iteration = 10

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
preprocess_layer = Preprocessing_Layer(mean,std)

pos = np.zeros(num_iteration // check_point)
epsilon = 16.0 / 255.0
step_size = 2.0 / 255 # PGD
# step_size = 1.6 / 255 # BIM

#### 加载模型，从本地加载
tnet = resnet152()
model_path = './torch_nets_weight/resnet152-imagenet.pth'
pre_dict = torch.load(model_path)
resnet_dict = tnet.state_dict()
state_dict = {k:v for k,v in pre_dict.items() if k in resnet_dict.keys()}
print("loaded pretrained weight. Len:",len(pre_dict.keys()),len(state_dict.keys()))
resnet_dict.update(state_dict)
model_dict = tnet.load_state_dict(resnet_dict)
tnet = nn.Sequential(preprocess_layer, tnet)
tnet = torch.nn.DataParallel(tnet, list(range(3))).cuda()
tnet.cuda()
tnet.eval()

# 加载学生网络
snet1 = models.resnet152(pretrained=False)
smodel1 = nn.Sequential(Normalize(mean=mean, std=std), snet1)
checkpoint1 = torch.load('./result/resnet152/20/1gpu_checkpoint.pth.tar')  # initial parameters of student model
load_pretrained_model(smodel1, checkpoint1['snet'])
smodel1.cuda()
smodel1 = torch.nn.DataParallel(smodel1, device_ids=[0,1,2])
smodel1.eval()

# 集成两个网络
print('ensemble')
ensemble = Ensemble([tnet, smodel1]).cuda()

model = ensemble

for i, (images, labels ,idx) in enumerate(data_loader):
    images = images.to(device)
    labels = labels.to(device)
    img = images.clone()
    for j in range(num_iteration):
        img_x = img
        img_x.requires_grad_(True)
        att_out = model(img_x)
        pred = torch.argmax(att_out, dim=1).view(-1)
        loss = nn.CrossEntropyLoss()(att_out, labels)
        model.zero_grad()
        loss.backward()
        input_grad = img_x.grad.data
        model.zero_grad()
        img = img.data + step_size * torch.sign(input_grad)
        img = torch.where(img > images + epsilon, images + epsilon, img)
        img = torch.where(img < images - epsilon, images - epsilon, img)
        img = torch.clamp(img, min=0, max=1)

        flag = (j+1) % check_point
        if flag == 0:
            point = j // check_point
            pos[point] = pos[point] + sum(torch.argmax(model(img),dim=1) != labels).cpu().numpy()
        if j == 9:
            save_images(img.detach().cpu().numpy(), img_list=data_set.img_path, idx=idx, output_dir=save_dir)
print(pos)