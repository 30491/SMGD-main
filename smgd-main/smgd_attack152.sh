#!/bin/bash

#CUDA_VISIBLE_DEVICES=0,1 python -u smgd_attack.py --output_dir=./output/resnet152/BIM/SGM_t --arch=resnet152 --momentum=0.0 --DI=0 --gamma=0.2
#CUDA_VISIBLE_DEVICES=0,1 python -u smgd_attack.py --output_dir=./output/resnet152/PGD/smgd_t --arch=resnet152 --momentum=1.0 --DI=0 --gamma=2.0
#CUDA_VISIBLE_DEVICES=0,1 python -u smgd_attack.py --output_dir=./output/resnet152/PGD/smgd_t --arch=resnet152 --momentum=0.0 --DI=1 --gamma=2.0
##CUDA_VISIBLE_DEVICES=0,1 python -u smgd_attack.py --output_dir=./output/resnet152/BIM/MI_DI_SGM_t --arch=resnet152 --momentum=1.0 --DI=1 --gamma=0.2
#CUDA_VISIBLE_DEVICES=0,1 python -u smgd_attack.py --output_dir=./output/resnet152/PGD/smgd_t --arch=resnet152 --momentum=1.0 --DI=1 --gamma=2.0
#CUDA_VISIBLE_DEVICES=0,1 python -u smgd_attack.py --output_dir=./output/resnet152/BIM/MI_SGM_t --arch=resnet152 --momentum=1.0 --DI=0 --gamma=0.2
#CUDA_VISIBLE_DEVICES=0,1 python -u smgd_attack.py --output_dir=./output/resnet152/BIM/DI_SGM_t --arch=resnet152 --momentum=0.0 --DI=1 --gamma=0.2

# 从2-40一共20个学生网络，开始集成攻击（每次只集成一个）
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t0_  --tem=2.0 --snet_dir1=./result/resnet152/2.0/1gpu_checkpoint.pth.tar
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t0_  --tem=4.0 --snet_dir1=./result/resnet152/4.0/1gpu_checkpoint.pth.tar
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t0_  --tem=6.0 --snet_dir1=./result/resnet152/6.0/1gpu_checkpoint.pth.tar
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t0_  --tem=8.0 --snet_dir1=./result/resnet152/8.0/1gpu_checkpoint.pth.tar
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t0_  --tem=10.0 --snet_dir1=./result/resnet152/10.0/1gpu_checkpoint.pth.tar
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t0_  --tem=12.0 --snet_dir1=./result/resnet152/12.0/1gpu_checkpoint.pth.tar
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t0_  --tem=14.0 --snet_dir1=./result/resnet152/14.0/1gpu_checkpoint.pth.tar
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t0_  --tem=16.0 --snet_dir1=./result/resnet152/16.0/1gpu_checkpoint.pth.tar

python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t0_  --tem=18.0 --snet_dir1=./result/resnet152/18/1gpu_checkpoint.pth.tar
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t0_  --tem=20.0 --snet_dir1=./result/resnet152/20/1gpu_checkpoint.pth.tar
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t0_  --tem=22.0 --snet_dir1=./result/resnet152/22/1gpu_checkpoint.pth.tar
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t0_  --tem=24.0 --snet_dir1=./result/resnet152/24/1gpu_checkpoint.pth.tar
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t0_  --tem=26.0 --snet_dir1=./result/resnet152/26/1gpu_checkpoint.pth.tar
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t0_  --tem=28.0 --snet_dir1=./result/resnet152/28/1gpu_checkpoint.pth.tar
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t0_  --tem=30.0 --snet_dir1=./result/resnet152/30/1gpu_checkpoint.pth.tar
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t0_  --tem=32.0 --snet_dir1=./result/resnet152/32/1gpu_checkpoint.pth.tar

python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t0_  --tem=34.0 --snet_dir1=./result/resnet152/34.0/1gpu_checkpoint.pth.tar
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t0_  --tem=36.0 --snet_dir1=./result/resnet152/36.0/1gpu_checkpoint.pth.tar
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t0_  --tem=38.0 --snet_dir1=./result/resnet152/38.0/1gpu_checkpoint.pth.tar
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t1_  --tem=40.0 --snet_dir1=./result/resnet152/40.0/1gpu_checkpoint.pth.tar




# 从一个学生开始蒸馏到8个学生 FGSM
#python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/FGSM/t_ --attack_method=fgsm --step_size=2 --ensemble_num=7 --batch_size=4
#python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/FGSM/t_ --attack_method=fgsm --step_size=2 --ensemble_num=6 --batch_size=4
#python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/FGSM/t_ --attack_method=fgsm --step_size=2 --ensemble_num=5 --batch_size=6
#python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/FGSM/t_ --attack_method=fgsm --step_size=2 --ensemble_num=4 --batch_size=8
#python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/FGSM/t_ --attack_method=fgsm --step_size=2 --ensemble_num=3 --batch_size=12
#python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/FGSM/t_ --attack_method=fgsm --step_size=2 --ensemble_num=2 --batch_size=15
#python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/FGSM/t_ --attack_method=fgsm --step_size=2 --ensemble_num=1 --batch_size=20
#
## 从一个学生开始蒸馏到8个学生 PGD
#python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t_ --attack_method=pgd --step_size=2 --ensemble_num=7 --batch_size=1
#python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t_ --attack_method=pgd --step_size=2 --ensemble_num=6 --batch_size=2
#python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t_ --attack_method=pgd --step_size=2 --ensemble_num=5 --batch_size=2
#python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t_ --attack_method=pgd --step_size=2 --ensemble_num=4 --batch_size=4
#python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t_ --attack_method=pgd --step_size=2 --ensemble_num=3 --batch_size=6
#python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t_ --attack_method=pgd --step_size=2 --ensemble_num=2 --batch_size=8
#python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t_ --attack_method=pgd --step_size=2 --ensemble_num=1 --batch_size=12

# 从一个学生开始蒸馏到8个学生 BIM
#python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/BIM/t_ --attack_method=pgd --step_size=-1 --ensemble_num=7 --batch_size=1
#python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/BIM/t_ --attack_method=pgd --step_size=-1 --ensemble_num=6 --batch_size=2
#python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/BIM/t_ --attack_method=pgd --step_size=-1 --ensemble_num=5 --batch_size=2
#python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/BIM/t_ --attack_method=pgd --step_size=-1 --ensemble_num=4 --batch_size=4
#python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/BIM/t_ --attack_method=pgd --step_size=-1 --ensemble_num=3 --batch_size=6
#python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/BIM/t_ --attack_method=pgd --step_size=-1 --ensemble_num=2 --batch_size=8
#python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/BIM/t_ --attack_method=pgd --step_size=-1 --ensemble_num=1 --batch_size=12

