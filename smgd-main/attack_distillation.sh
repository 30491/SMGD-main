#!/bin/bash

Tempetarure=(8 7 6 5 4 3 2 1)

clear

for tem in "${Tempetarure[@]}"
do
  CUDA_VISIBLE_DEVICES=0,1,2 python3 -u attack_resnet152.py --ensemble_num=$tem
done
