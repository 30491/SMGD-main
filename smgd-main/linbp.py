import torch
import argparse
import os
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision import transforms
from utils_data import SubsetImageNet, save_images
import pretrainedmodels
import pretrainedmodels.utils

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

## LinBP utils
def linbp_forw_resnet50(model, x, do_linbp, linbp_layer):
    jj = int(linbp_layer.split('_')[0])
    kk = int(linbp_layer.split('_')[1])
    x = model[0](x)
    x = model[1].conv1(x)
    x = model[1].bn1(x)
    x = model[1].relu(x)
    x = model[1].maxpool(x)
    ori_mask_ls = []
    conv_out_ls = []
    relu_out_ls = []
    conv_input_ls = []
    def layer_forw(jj, kk, jj_now, kk_now, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls, do_linbp):
        if jj < jj_now:
            x, ori_mask, conv_out, relu_out, conv_in = block_func(mm, x, linbp=True)
            ori_mask_ls.append(ori_mask)
            conv_out_ls.append(conv_out)
            relu_out_ls.append(relu_out)
            conv_input_ls.append(conv_in)
        elif jj == jj_now:
            if kk_now >= kk:
                x, ori_mask, conv_out, relu_out, conv_in = block_func(mm, x, linbp=True)
                ori_mask_ls.append(ori_mask)
                conv_out_ls.append(conv_out)
                relu_out_ls.append(relu_out)
                conv_input_ls.append(conv_in)
            else:
                x, _, _, _, _ = block_func(mm, x, linbp=False)
        else:
            x, _, _, _, _ = block_func(mm, x, linbp=False)
        return x, ori_mask_ls
    for ind, mm in enumerate(model[1].layer1):
        x, ori_mask_ls = layer_forw(jj, kk, 1, ind, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls, do_linbp)
    for ind, mm in enumerate(model[1].layer2):
        x, ori_mask_ls = layer_forw(jj, kk, 2, ind, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls, do_linbp)
    for ind, mm in enumerate(model[1].layer3):
        x, ori_mask_ls = layer_forw(jj, kk, 3, ind, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls, do_linbp)
    for ind, mm in enumerate(model[1].layer4):
        x, ori_mask_ls = layer_forw(jj, kk, 4, ind, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls, do_linbp)
    x = model[1].avgpool(x)
    x = torch.flatten(x, 1)
    x = model[1].fc(x)
    return x, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls

def block_func(block, x, linbp):
    identity = x
    conv_in = x+0
    out = block.conv1(conv_in)
    out = block.bn1(out)
    out_0 = out + 0
    if linbp:
        out = linbp_relu(out_0)
    else:
        out = block.relu(out_0)
    ori_mask_0 = out.data.bool().int()

    out = block.conv2(out)
    out = block.bn2(out)
    out_1 = out + 0
    if linbp:
        out = linbp_relu(out_1)
    else:
        out = block.relu(out_1)
    ori_mask_1 = out.data.bool().int()

    out = block.conv3(out)
    out = block.bn3(out)

    if block.downsample is not None:
        identity = block.downsample(identity)
    identity_out = identity + 0
    x_out = out + 0


    out = identity_out + x_out
    out = block.relu(out)
    ori_mask_2 = out.data.bool().int()
    return out, (ori_mask_0, ori_mask_1, ori_mask_2), (identity_out, x_out), (out_0, out_1), (0, conv_in)

def linbp_relu(x):
    x_p = F.relu(-x)
    x = x + x_p.data
    return x

def linbp_backw_resnet50(img, loss, conv_out_ls, ori_mask_ls, relu_out_ls, conv_input_ls, xp):
    for i in range(-1, -len(conv_out_ls)-1, -1):
        if i == -1:
            grads = torch.autograd.grad(loss, conv_out_ls[i])
        else:
            grads = torch.autograd.grad((conv_out_ls[i+1][0], conv_input_ls[i+1][1]), conv_out_ls[i], grad_outputs=(grads[0], main_grad_norm))
        normal_grad_2 = torch.autograd.grad(conv_out_ls[i][1], relu_out_ls[i][1], grads[1]*ori_mask_ls[i][2],retain_graph=True)[0]
        normal_grad_1 = torch.autograd.grad(relu_out_ls[i][1], relu_out_ls[i][0], normal_grad_2 * ori_mask_ls[i][1], retain_graph=True)[0]
        normal_grad_0 = torch.autograd.grad(relu_out_ls[i][0], conv_input_ls[i][1], normal_grad_1 * ori_mask_ls[i][0], retain_graph=True)[0]
        del normal_grad_2, normal_grad_1
        main_grad = torch.autograd.grad(conv_out_ls[i][1], conv_input_ls[i][1], grads[1])[0]
        alpha = normal_grad_0.norm(p=2, dim = (1,2,3), keepdim = True) / main_grad.norm(p=2,dim = (1,2,3), keepdim=True)
        main_grad_norm = xp * alpha * main_grad
    input_grad = torch.autograd.grad((conv_out_ls[0][0], conv_input_ls[0][1]), img, grad_outputs=(grads[0], main_grad_norm))
    return input_grad[0].data


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
parser.add_argument('--output_dir', default='./output/resnet50/BIM/1Lin_t_', help='the path of the saved dataset')
parser.add_argument('--arch', default='resnet50',help='source model for black-box attack evaluation',choices=model_names)

parser.add_argument('--ensemble', default=1, type=int) # 1为True
parser.add_argument('--ensemble_num', default='1', type=int) ### 集成网络的个数，到底集成几个学生网络 每次实验前都检查学生网路权重
parser.add_argument('--snet_dir1', default='./result/resnet50/20/1gpu_checkpoint.pth.tar', help='the path of snet1') ## batchsize = 30
# parser.add_argument('--snet_dir2', default='./result/densenet201/22/1gpu_checkpoint.pth.tar', help='the path of snet2')
# parser.add_argument('--snet_dir3', default='./result/densenet201/24/1gpu_checkpoint.pth.tar', help='the path of snet3')
# parser.add_argument('--snet_dir4', default='./result/densenet201/26/1gpu_checkpoint.pth.tar', help='the path of snet4')
# parser.add_argument('--snet_dir5', default='./result/densenet201/28/1gpu_checkpoint.pth.tar', help='the path of snet5') ## batchsize = 9
# parser.add_argument('--snet_dir6', default='./result/densenet201/30/1gpu_checkpoint.pth.tar', help='the path of snet6')
# parser.add_argument('--snet_dir7', default='./result/densenet201/32/1gpu_checkpoint.pth.tar', help='the path of snet7') ## batchsize = 6
# parser.add_argument('--snet_dir8', default='./result/densenet201/18/1gpu_checkpoint.pth.tar', help='the path of snet8')
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
data_loader = torch.utils.data.DataLoader(data_set, batch_size=20, shuffle=False, **kwargs)

## 图片保存的地方
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
tnet = models.resnet50(pretrained=True)
tmodel = nn.Sequential(preprocess_layer, tnet)
tmodel.cuda()
tmodel.eval()

# tnet = models.__dict__[args.arch](pretrained= True)
# tmodel = nn.Sequential(Normalize(mean=mean, std=std), tnet)
# tmodel.cuda()
# tmodel = torch.nn.DataParallel(tmodel,device_ids=[0,1,2])
# tmodel.eval()

# 加载学生网络
if args.ensemble == True:
    ####################################
    ### student model
    # ### 加载第一个学生网络
    snet1 = models.resnet50(pretrained=False)
    smodel1 = nn.Sequential(preprocess_layer, snet1)
    checkpoint1 = torch.load(args.snet_dir1)  # initial parameters of student model
    load_pretrained_model(smodel1, checkpoint1['snet'])
    smodel1.cuda()
    smodel1.eval()
    #################################### 集成网络

    print('ensemble_num',ensemble_num)
    if ensemble_num == 1:
        print('ensemble_num = ', ensemble_num)
        ensemble = Ensemble([tmodel, smodel1]).to(device)

    print("--ensemble success--")


if args.ensemble == True:
    model  = ensemble
else:
    model = tmodel


linbp_layer = '3_1'
sgm_lambda = 1.0
for i, (images, labels ,idx) in enumerate(data_loader):
    images = images.to(device)
    labels = labels.to(device)
    img = images.clone()
    for j in range(num_iteration):
        img_x = img
        img_x.requires_grad_(True)
        att_out, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls = linbp_forw_resnet50(model, img_x, True,
                                                                                            linbp_layer)
        pred = torch.argmax(att_out, dim=1).view(-1)
        loss = nn.CrossEntropyLoss()(att_out, labels)
        model.zero_grad()
        input_grad = linbp_backw_resnet50(img_x, loss, conv_out_ls, ori_mask_ls, relu_out_ls, conv_input_ls,
                                          xp=sgm_lambda)

        model.zero_grad()

        img = img.data + step_size * torch.sign(input_grad)
        img = torch.where(img > images + epsilon, images + epsilon, img)
        img = torch.where(img < images - epsilon, images - epsilon, img)
        img = torch.clamp(img, min=0, max=1)

        flag = (j + 1) % check_point
        if flag == 0:
            point = j // check_point
            pos[point] = pos[point] + sum(torch.argmax(model(img), dim=1) != labels).cpu().numpy()
        if j == 9:
            save_images(img.detach().cpu().numpy(), img_list=data_set.img_path, idx=idx, output_dir=output_dir)
print(pos)