import argparse
import torch
import torch.nn as nn
from torchvision import transforms
import time
import torchvision.models as models
import os
# os.environ['CUDA_VISIBLE_DEVICES'] ="0"
import numpy as np
from utils import load_pretrained_model
import pretrainedmodels
from utils_sgm import register_hook_for_resnet, register_hook_for_densenet
from utils_data import SubsetImageNet, save_images
from attack_method import FGSMAttack,PGDAttack

model_names = sorted(name for name in pretrainedmodels.__dict__
                     if not name.startswith("__")
                     and name.islower()
                     and callable(pretrainedmodels.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Attack Evaluation')#

parser.add_argument('--input_dir', default='./dataset/images1000', help='the path of original dataset')

parser.add_argument('--output_dir', default='./output/resnet152/PGD/1000t_', help='the path of the saved dataset')
parser.add_argument('--attack_method', default='pgd', type=str,choices=['fgsm','pgd'])
parser.add_argument('--batch_size', type=int, default=20, metavar='N',help='input batch size for adversarial attack')
parser.add_argument('--ensemble', default=1, type=int) # 1为True
parser.add_argument("--DI", type=int, default=0) #
parser.add_argument("--TI", type=int, default=0) # 必须结合MI，才是TI
parser.add_argument('--momentum', default=0.0, type=float) # 1,就是MI；不用就是0.0
parser.add_argument('--gamma', default=2.0, type=float) # SGM: FGSM,0.5-Resnet,0.7-Dense;PGD,0.2-Resnet,0.5-Dense；不使用就是2.0
parser.add_argument("--amplification", type=float, default=0.0, help="To amplifythe step size.") ## 不用就是0.0    PI,10;+SMGD,5;加所有组合，2.5 ,gamma再乘0.5
parser.add_argument("--pi_size", type=int, default=3, help="k size")

parser.add_argument('--ensemble_num', default='1', type=int) ### 集成网络的个数，到底集成几个学生网络 每次实验前都检查学生网路权重
parser.add_argument('--arch', default='resnet152',help='source model for black-box attack evaluation',choices=model_names)
parser.add_argument('--snet_dir1', default='./result/resnet152/images1000/20/1gpu_checkpoint.pth.tar', help='the path of snet1') ## batchsize = 30
parser.add_argument('--snet_dir2', default='./result/resnet152/22/1gpu_checkpoint.pth.tar', help='the path of snet2')
parser.add_argument('--snet_dir3', default='./result/resnet152/24/1gpu_checkpoint.pth.tar', help='the path of snet3')
parser.add_argument('--snet_dir4', default='./result/resnet152/26/1gpu_checkpoint.pth.tar', help='the path of snet4')
parser.add_argument('--snet_dir5', default='./result/resnet152/28/1gpu_checkpoint.pth.tar', help='the path of snet5') ## batchsize = 9
parser.add_argument('--snet_dir6', default='./result/resnet152/30/1gpu_checkpoint.pth.tar', help='the path of snet6')
parser.add_argument('--snet_dir7', default='./result/resnet152/32/1gpu_checkpoint.pth.tar', help='the path of snet7') ## batchsize = 6
parser.add_argument('--snet_dir8', default='./result/resnet152/18/1gpu_checkpoint.pth.tar', help='the path of snet8')
# parser.add_argument('--snet_dir9', default='./result/resnet152/21/1gpu_checkpoint.pth.tar', help='the path of snet9')
# parser.add_argument('--snet_dir10', default='./result/resnet152/23/1gpu_checkpoint.pth.tar', help='the path of snet10')
####################
parser.add_argument('--cuda', type=int, default=[1,2,3])
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--epsilon', default=16, type=float, help='perturbation')
parser.add_argument('--num_steps', default=10, type=int, help='perturb number of steps')
parser.add_argument('--step_size', default=2, type=float, help='perturb step size') ##  2 = pgd; -1 = bim

parser.add_argument('--print_freq', default=10, type=int)

parser.add_argument("--prob", type=float, default=0.0, help="probability of using diverse inputs.")
parser.add_argument("--image_width", type=int, default=224, help="Width of each input images.")
parser.add_argument("--image_height", type=int, default=224, help="Height of each input images.")
parser.add_argument("--image_resize", type=int, default=247, help="Resize of each input images.")

args = parser.parse_args()
# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

ensemble_num = args.ensemble_num

output_dir = args.output_dir + str(ensemble_num)
# print(output_dir)
if os.path.exists(str(output_dir)) == False:
    os.makedirs(str(output_dir))


# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]

def generate_adversarial_example(model, data_loader, adversary, img_path):
    """
    evaluate model by black-box attack
    """
    model.eval()

    for batch_idx, (inputs, true_class, idx) in enumerate(data_loader):

        inputs, true_class = \
            inputs.to(device), true_class.to(device)

        inputs_adv = adversary.perturb(inputs, true_class)
        save_images(inputs_adv.detach().cpu().numpy(), img_list=img_path,
                    idx=idx, output_dir=output_dir)
        # assert False
        if batch_idx % args.print_freq == 0:
            print('generating: [{0}/{1}]'.format(batch_idx, len(data_loader)))

import torch.nn.functional as F
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

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

def main():
    print(output_dir) # 对抗样本输出位置

    begin=time.time()
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    data_set = SubsetImageNet(root=args.input_dir, transform=transform_test)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=False, **kwargs)

    # create models teacher
    # net = pretrainedmodels.__dict__[args.arch](num_classes=1000, pretrained='imagenet')  # densenet 121
    tnet = models.__dict__[args.arch](pretrained= True)
    tmodel = nn.Sequential(Normalize(mean=mean, std=std), tnet)
    tmodel.cuda()
    tmodel = torch.nn.DataParallel(tmodel,device_ids=[0,1,2])

    # tmodel = smodel1
    tmodel.eval()



    if args.ensemble == True:
        ####################################
        # net2 = pretrainedmodels.__dict__[args.arch](num_classes=1000, pretrained='imagenet')
        ### student model
        # ### 加载第一个学生网络
        snet1 = models.__dict__[args.arch](pretrained=False)
        smodel1 = nn.Sequential(Normalize(mean=mean, std=std), snet1)
        checkpoint1 = torch.load(args.snet_dir1)  # initial parameters of student model
        load_pretrained_model(smodel1, checkpoint1['snet'])
        smodel1.cuda()
        smodel1 = torch.nn.DataParallel(smodel1, device_ids=[0,1,2])

        smodel1.eval()
        # snet2
        snet2 = models.__dict__[args.arch](pretrained=False)
        smodel2 = nn.Sequential(Normalize(mean=mean, std=std), snet2)
        checkpoint2 = torch.load(args.snet_dir2)  # initial parameters of student model
        load_pretrained_model(smodel2, checkpoint2['snet'])
        smodel2.cuda()
        smodel2 = torch.nn.DataParallel(smodel2, device_ids=[0,1,2])

        smodel2.eval()
        # ### snet3
        snet3 = models.__dict__[args.arch](pretrained=False)
        smodel3 = nn.Sequential(Normalize(mean=mean, std=std), snet3)
        checkpoint3 = torch.load(args.snet_dir3)  # initial parameters of student model
        load_pretrained_model(smodel3, checkpoint3['snet'])
        smodel3.cuda()
        smodel3 = torch.nn.DataParallel(smodel3, device_ids=[0,1,2])
        smodel3.eval()
        # ### snet4
        snet4 = models.__dict__[args.arch](pretrained=False)
        smodel4 = nn.Sequential(Normalize(mean=mean, std=std), snet4)
        checkpoint4 = torch.load(args.snet_dir4)  # initial parameters of student model
        load_pretrained_model(smodel4, checkpoint4['snet'])
        smodel4.cuda()
        smodel4 = torch.nn.DataParallel(smodel4, device_ids=[0,1,2])
        smodel4.eval()
        # ### snet5
        snet5 = models.__dict__[args.arch](pretrained=False)
        smodel5 = nn.Sequential(Normalize(mean=mean, std=std), snet5)
        checkpoint5 = torch.load(args.snet_dir5)  # initial parameters of student model
        load_pretrained_model(smodel5, checkpoint5['snet'])
        smodel5.cuda()
        smodel5 = torch.nn.DataParallel(smodel5, device_ids=[0,1,2])
        smodel5.eval()
         ### snet6
        snet6 = models.__dict__[args.arch](pretrained=False)
        smodel6 = nn.Sequential(Normalize(mean=mean, std=std), snet6)
        checkpoint6 = torch.load(args.snet_dir6)  # initial parameters of student model
        load_pretrained_model(smodel6, checkpoint6['snet'])
        smodel6.cuda()
        smodel6 = torch.nn.DataParallel(smodel6, device_ids=[0,1,2])
        smodel6.eval()
        # # # ### snet7
        snet7 = models.__dict__[args.arch](pretrained=False)
        smodel7 = nn.Sequential(Normalize(mean=mean, std=std), snet7)
        checkpoint7 = torch.load(args.snet_dir7)  # initial parameters of student model
        load_pretrained_model(smodel7, checkpoint7['snet'])
        smodel7.cuda()
        smodel7 = torch.nn.DataParallel(smodel7, device_ids=[0,1,2])
        smodel7.eval()
        # # ### snet8
        snet8 = models.__dict__[args.arch](pretrained=False)
        smodel8 = nn.Sequential(Normalize(mean=mean, std=std), snet8)
        checkpoint8 = torch.load(args.snet_dir8)  # initial parameters of student model
        load_pretrained_model(smodel8, checkpoint8['snet'])
        smodel8.cuda()
        smodel8 = torch.nn.DataParallel(smodel8, device_ids=[0,1,2])
        smodel8.eval()


        #################################### 集成网络

        # print(ensemble_num)
        # print(type(ensemble_num))
        # exit()
        if ensemble_num == 1 :
            print('ensemble_num = ',ensemble_num)
            ensemble=Ensemble([tmodel, smodel1]).to(device)
        if ensemble_num == 2 :
            print('ensemble_num = ',ensemble_num)
            ensemble=Ensemble([tmodel, smodel1, smodel2]).to(device)
        if ensemble_num == 3 :
            print('ensemble_num = ',ensemble_num)
            ensemble=Ensemble([tmodel, smodel1, smodel2, smodel3]).to(device)
        if ensemble_num == 4 :
            print('ensemble_num = ',ensemble_num)
            ensemble=Ensemble([tmodel, smodel1, smodel2, smodel3, smodel4]).to(device)
        if ensemble_num == 5 :
            print('ensemble_num = ',ensemble_num)
            ensemble=Ensemble([tmodel, smodel1, smodel2, smodel3 ,smodel4, smodel5]).to(device)
        if ensemble_num == 6:
            print('ensemble_num = ', ensemble_num)
            ensemble = Ensemble([tmodel, smodel1, smodel2, smodel3, smodel4, smodel5, smodel6]).to(device)
        if ensemble_num == 7 :
            print('ensemble_num = ',ensemble_num)
            ensemble=Ensemble([tmodel, smodel1, smodel2, smodel3 ,smodel4, smodel5, smodel6, smodel7]).to(device)
        if ensemble_num == 8 :
            print('ensemble_num = ',ensemble_num)
            ensemble=Ensemble([tmodel, smodel1, smodel2, smodel3 ,smodel4, smodel5,smodel6, smodel7,smodel8]).to(device)
        # if ensemble_num == 9 :
        #     print('ensemble_num = ',ensemble_num)
        #     ensemble=Ensemble([tmodel, smodel1, smodel2, smodel3 ,smodel4, smodel5,smodel6, smodel7,smodel8,smodel9]).to(device)
        # if ensemble_num == 10 :
        #     print('ensemble_num = ',ensemble_num)
        #     ensemble=Ensemble([tmodel, smodel1, smodel2, smodel3 ,smodel4, smodel5,smodel6, smodel7,smodel8,smodel9,smodel10]).to(device)

        print("--ensemble success--")
    epsilon = args.epsilon / 255.0 # 16
    if args.step_size < 0:
        step_size = epsilon / args.num_steps  # 16/10
        print("<0")
    else:
        step_size = args.step_size / 255.0  # 2

    # using our method - Skip Gradient Method (SGM)
    if args.gamma < 1.0:
        print('using SGM Method')
        if args.arch in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            register_hook_for_resnet(tmodel, arch=args.arch, gamma=args.gamma)
            if args.ensemble==True:
                register_hook_for_resnet(smodel1, arch=args.arch, gamma=args.gamma)
        elif args.arch in ['densenet121', 'densenet169', 'densenet201']:
            register_hook_for_densenet(tmodel, arch=args.arch, gamma=args.gamma)
            if args.ensemble == True:
                register_hook_for_densenet(smodel1, arch=args.arch, gamma=args.gamma)
        else:
            raise ValueError('Current code only supports resnet/densenet. '
                             'You can extend this code to other architectures.')

    if args.momentum > 0.0:
        print('-- momentum --')

    if args.DI == True:
        print("input diversity load")
        args.prob = 0.7
    if args.ensemble == True:
        print('-- ensemble loss --')
        loss = nn.NLLLoss(reduction='mean').to(device)
        if args.TI == True:
            if args.attack_method == 'fgsm':
                print('using FGSM_TIM attack  -- 1')
                adversary = FGSMAttack(predict=ensemble, loss_fn=loss,
                                              eps=epsilon, clip_min=0.0, clip_max=1.0, targeted=False,ti_size=15,image_width=224, image_resize=247, prob=args.prob)
            if args.attack_method == 'pgd':
                print('using PGD_TIM attack   --2')
                adversary = PGDAttack(model=ensemble, epsilon=epsilon,num_steps=args.num_steps,step_size=step_size,
                                      random_start=False,image_width=224, image_resize=247, prob=args.prob, momentum=args.momentum,
                                      ti_size=15, loss_fn=loss, targeted=False,pi_amplification=args.amplification,pi_size=args.pi_size)
        else:
            if args.attack_method=='fgsm':
                print('using linf FGSM attack --3 ')  #
                print('fgsm ensemble')
                adversary = FGSMAttack(predict=ensemble, loss_fn=loss,
                                     eps=epsilon, clip_min=0.0, clip_max=1.0, targeted=False, ti_size=1,image_width=224, image_resize=247, prob=args.prob)
            if args.attack_method=='pgd':
                print('using linf PGD attack  --4')
                adversary = PGDAttack(model=ensemble, epsilon=epsilon, num_steps=args.num_steps, step_size=step_size,
                                      random_start=False, image_width=224, image_resize=247, prob=args.prob, momentum=args.momentum,
                                      ti_size=1, loss_fn=loss, targeted=False,pi_amplification=args.amplification,pi_size=args.pi_size)

    else:
        print('no  ensemble')
        loss = nn.CrossEntropyLoss(reduction="sum")
        if args.TI==True:

            if args.attack_method == 'fgsm':
                print('using FGSM_TIM attack  --5')
                adversary = FGSMAttack(predict=tmodel, loss_fn=loss,
                                              eps=epsilon, clip_min=0.0, clip_max=1.0, targeted=False,ti_size=15, image_width=224, image_resize=247,prob=args.prob)
            if args.attack_method == 'pgd':
                print('using PGD_TIM attack  --6')
                adversary = PGDAttack(model=tmodel, epsilon=epsilon,num_steps=args.num_steps,step_size=step_size,
                                      random_start=False,image_width=224, image_resize=247, prob=args.prob, momentum=args.momentum,
                                      ti_size=15, loss_fn=loss, targeted=False,pi_amplification=args.amplification,pi_size=args.pi_size)

        else:
            if args.attack_method == 'fgsm':
                print('using linf FGSM attack --7')
                adversary = FGSMAttack(predict=tmodel, loss_fn=loss,
                                     eps=epsilon, clip_min=0.0, clip_max=1.0, targeted=False, ti_size=1, image_width=224, image_resize=247,prob=args.prob)
            if args.attack_method == 'pgd':
                print('using linf PGD attack --8')
                adversary = PGDAttack(model=tmodel, epsilon=epsilon, num_steps=args.num_steps, step_size=step_size,
                                      random_start=False, image_width=224, image_resize=247, prob=args.prob,momentum=args.momentum,
                                      ti_size=1, loss_fn=loss, targeted=False,pi_amplification=args.amplification,pi_size=args.pi_size)

    generate_adversarial_example(model=tmodel, data_loader=data_loader,
                                 adversary=adversary, img_path=data_set.img_path)

    end=time.time()
    print("time:",end-begin)
if __name__ == '__main__':
    main()
