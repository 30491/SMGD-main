#!/bin/bash

Model=(resnet152 resnet101 resnet50 densenet201 senet154 vgg19_bn inceptionv3 inceptionv4 inceptionresnetv2)

#AdvPath=('./output/resnet152/PGD/t_8')
#AdvPath1=('/home/liuminglin/zhangpan/NRP/NRP_attack_densenet201_FGSM_si_t_0')




clear

for model in "${Model[@]}"
do
  CUDA_VISIBLE_DEVICES=1 python3.6 -u evaluate.py --input_dir=./output/densenet201/BIM/DI_TI_t_0  --arch=$model --number='DI_TI_t_0'
done

#clear
#
#for model in "${Model[@]}"
#do
#  CUDA_VISIBLE_DEVICES=1 python3.6 -u evaluate.py --input_dir=./output/resnet152/PGD/t0_34.0  --arch=$model --number='0_34.0'
#done
#
#clear
#
#for model in "${Model[@]}"
#do
#  CUDA_VISIBLE_DEVICES=1 python3.6 -u evaluate.py --input_dir=./output/resnet152/PGD/t0_36.0  --arch=$model --number='0_36.0'
#done
#
#clear
#
#for model in "${Model[@]}"
#do
#  CUDA_VISIBLE_DEVICES=1 python3.6 -u evaluate.py --input_dir=./output/resnet152/PGD/t0_38.0  --arch=$model --number='0_38.0'
#done
#
#clear
#
#for model in "${Model[@]}"
#do
#  CUDA_VISIBLE_DEVICES=1 python3.6 -u evaluate.py --input_dir=./output/resnet152/PGD/t0_40.0  --arch=$model --number='0_40.0'
#done