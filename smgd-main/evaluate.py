# -*- coding: UTF-8 -*-
from __future__ import print_function, division, absolute_import
import argparse
import os
import time
import sys
from utils import load_pretrained_model
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import logging
import csv
import pretrainedmodels
import pretrainedmodels.utils
import pandas as pd
import torchvision.models as models
from utils_data import SubsetImageNet#

model_names = sorted(name for name in pretrainedmodels.__dict__
                     if not name.startswith("__")
                     and name.islower()
                     and callable(pretrainedmodels.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--input_dir', metavar='DIR',
                    # default="./dataset/SuFGSMageNet224/",
                    default="/home/liuminglin/zhangpan/NRP/NRP_attack_densenet201_BIM_TI_t_0",
                    help='path to dataset')
parser.add_argument('-b', '--batch-size', default=200, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--number', default='NRP_TI_t_0', type=str)
parser.add_argument('--arch', '-a', metavar='ARCH',
                    default='resnet152',
                    # default='resnet101',
                    # default='resnet50',
                    # default='densenet201',
                    # default='senet154',
                    # default='vgg19_bn',
                    # default='inceptionv3',
                    # default='inceptionv4',
                    # default='inceptionresnetv2',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) + ' (default: fbresnet152)')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', default='imagenet', help='use pre-trained model')
parser.add_argument('--do-not-preserve-aspect-ratio',
                    dest='preserve_aspect_ratio',
                    help='do not preserve the aspect ratio when resizing an image',
                    action='store_false')
parser.add_argument('--csv', type=int, default=1, help='') # 1 = True
args = parser.parse_args()

parser.set_defaults(preserve_aspect_ratio=True)
best_prec1 = 0
number = args.number

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)

path = './output/densenet201/BIM/' # 存储路径，保存txt和excel文件
if os.path.exists(path) == False:
    os.makedirs(path)
fh = logging.FileHandler(os.path.join(path + 'log'+str(number)+'.txt'))  ### 实验数据保存位置
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

args,unparsed=parser.parse_known_args()
a = str(args.arch)
b = str(args.input_dir)
logging.info(a)
logging.info(b)


def main():
    global args, best_prec1
    args = parser.parse_args()


    if args.pretrained.lower() not in ['false', 'none', 'not', 'no', '0']:

        model = pretrainedmodels.__dict__[args.arch](num_classes=1000,pretrained=args.pretrained)
        # model = models.__dict__[args.arch](pretrained=True)



    else:
        model = pretrainedmodels.__dict__[args.arch]()


    # Data loading code
    valdir = os.path.join(args.input_dir)

    scale = 1.0
    val_tf = pretrainedmodels.utils.TransformImage(
        model,
        scale=scale,
        preserve_aspect_ratio=args.preserve_aspect_ratio
    )
    val_set = SubsetImageNet(root=valdir, transform=val_tf)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    model = model.cuda()


    validate(val_loader, model, criterion)


def validate(val_loader, model, criterion):
    with torch.no_grad():
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, raw_data in enumerate(val_loader):
            input = raw_data[0]
            target = raw_data[1]
            target = target.cuda()
            input = input.cuda()

            output = model(input)
            output = output.contiguous()
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if (i+1) % args.print_freq == 0:
            #     print('Test: [{0}/{1}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            #           'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            #            i, len(val_loader), batch_time=batch_time, loss=losses,
            #            top1=top1, top5=top5))

        print('* Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}\n'
              '* attack@1 {attack1:.3f} attack@5 {attack5:.3f}'
              .format(top1=top1, top5=top5,attack1=100-top1.avg,attack5=100-top5.avg))
        a = 100 - top1.avg
        logging.info('* attack@1 {attack1:.3f} attack@5 {attack5:.3f}'
              .format(attack1=100 - top1.avg, attack5=100 - top5.avg))
        logging.info(' {attack1:.3f}/{attack5:.3f}'
              .format(attack1=100 - top1.avg, attack5=100 - top5.avg))
        logging.info("================================================")


        if args.csv==True:
            if args.arch=='resnet152':
                with open(os.path.join(path, 'success'+str(number)+'.csv'), mode='w', encoding="utf-8-sig", newline="") as f:
                    csv_write = csv.writer(f)
                    csv_write.writerow([args.arch])
                    csv_write.writerow([round((100 - top1.avg),2)])
            else:
                data=pd.read_csv(os.path.join(path,'success'+str(number)+'.csv'),float_precision="round_trip")
                data1=[round((100 - top1.avg),2)]
                data[args.arch]=data1
                data.to_csv(os.path.join(path,'success'+str(number)+'.csv'),mode='w',index=False)

        return top1.avg, top5.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = correct.contiguous()
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()