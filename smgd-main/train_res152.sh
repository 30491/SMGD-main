#!/bin/bash

Model=(resnet152)

T=(6 8 10 12 14 16 34 36 38 40)

clear
for t in "${T[@]}"
do
  for model in "${Model[@]}"
  do
    python3.6 -u train_res152.py --T=$t --arch=$model
  done
done

python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t1_  --tem=2.0 --snet_dir1=./result/resnet152/2.0/1gpu_checkpoint.pth.tar
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t1_  --tem=4.0 --snet_dir1=./result/resnet152/4.0/1gpu_checkpoint.pth.tar
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t1_  --tem=6.0 --snet_dir1=./result/resnet152/6.0/1gpu_checkpoint.pth.tar
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t1_  --tem=8.0 --snet_dir1=./result/resnet152/8.0/1gpu_checkpoint.pth.tar
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t1_  --tem=10.0 --snet_dir1=./result/resnet152/10.0/1gpu_checkpoint.pth.tar
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t1_  --tem=12.0 --snet_dir1=./result/resnet152/12.0/1gpu_checkpoint.pth.tar
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t1_  --tem=14.0 --snet_dir1=./result/resnet152/14.0/1gpu_checkpoint.pth.tar
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t1_  --tem=16.0 --snet_dir1=./result/resnet152/16.0/1gpu_checkpoint.pth.tar

python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t1_  --tem=18.0 --snet_dir1=./result/resnet152/18/1gpu_checkpoint.pth.tar
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t1_  --tem=20.0 --snet_dir1=./result/resnet152/20/1gpu_checkpoint.pth.tar
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t1_  --tem=22.0 --snet_dir1=./result/resnet152/22/1gpu_checkpoint.pth.tar
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t1_  --tem=24.0 --snet_dir1=./result/resnet152/24/1gpu_checkpoint.pth.tar
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t1_  --tem=26.0 --snet_dir1=./result/resnet152/26/1gpu_checkpoint.pth.tar
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t1_  --tem=28.0 --snet_dir1=./result/resnet152/28/1gpu_checkpoint.pth.tar
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t1_  --tem=30.0 --snet_dir1=./result/resnet152/30/1gpu_checkpoint.pth.tar
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t1_  --tem=32.0 --snet_dir1=./result/resnet152/32/1gpu_checkpoint.pth.tar

python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t1_  --tem=34.0 --snet_dir1=./result/resnet152/34.0/1gpu_checkpoint.pth.tar
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t1_  --tem=36.0 --snet_dir1=./result/resnet152/36.0/1gpu_checkpoint.pth.tar
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t1_  --tem=38.0 --snet_dir1=./result/resnet152/38.0/1gpu_checkpoint.pth.tar
python3.6 -u smgd_attack152.py --output_dir=./output/resnet152/PGD/t1_  --tem=40.0 --snet_dir1=./result/resnet152/40.0/1gpu_checkpoint.pth.tar