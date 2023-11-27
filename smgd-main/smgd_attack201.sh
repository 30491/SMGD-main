#!/bin/bash

# 从一个学生开始蒸馏到8个学生 FGSM
python3.6 -u smgd_attack201.py --output_dir=./output/densenet201/FGSM/t_ --attack_method=fgsm --step_size=2 --ensemble_num=7 --batch_size=8
python3.6 -u smgd_attack201.py --output_dir=./output/densenet201/FGSM/t_ --attack_method=fgsm --step_size=2 --ensemble_num=6 --batch_size=10
python3.6 -u smgd_attack201.py --output_dir=./output/densenet201/FGSM/t_ --attack_method=fgsm --step_size=2 --ensemble_num=5 --batch_size=12
python3.6 -u smgd_attack201.py --output_dir=./output/densenet201/FGSM/t_ --attack_method=fgsm --step_size=2 --ensemble_num=4 --batch_size=12
python3.6 -u smgd_attack201.py --output_dir=./output/densenet201/FGSM/t_ --attack_method=fgsm --step_size=2 --ensemble_num=3 --batch_size=15
python3.6 -u smgd_attack201.py --output_dir=./output/densenet201/FGSM/t_ --attack_method=fgsm --step_size=2 --ensemble_num=2 --batch_size=20
python3.6 -u smgd_attack201.py --output_dir=./output/densenet201/FGSM/t_ --attack_method=fgsm --step_size=2 --ensemble_num=1 --batch_size=25

# 从一个学生开始蒸馏到8个学生 PGD
python3.6 -u smgd_attack201.py --output_dir=./output/densenet201/PGD/t_ --attack_method=pgd --step_size=2 --ensemble_num=7 --batch_size=4
python3.6 -u smgd_attack201.py --output_dir=./output/densenet201/PGD/t_ --attack_method=pgd --step_size=2 --ensemble_num=6 --batch_size=4
python3.6 -u smgd_attack201.py --output_dir=./output/densenet201/PGD/t_ --attack_method=pgd --step_size=2 --ensemble_num=5 --batch_size=6
python3.6 -u smgd_attack201.py --output_dir=./output/densenet201/PGD/t_ --attack_method=pgd --step_size=2 --ensemble_num=4 --batch_size=6
python3.6 -u smgd_attack201.py --output_dir=./output/densenet201/PGD/t_ --attack_method=pgd --step_size=2 --ensemble_num=3 --batch_size=8
python3.6 -u smgd_attack201.py --output_dir=./output/densenet201/PGD/t_ --attack_method=pgd --step_size=2 --ensemble_num=2 --batch_size=10
python3.6 -u smgd_attack201.py --output_dir=./output/densenet201/PGD/t_ --attack_method=pgd --step_size=2 --ensemble_num=1 --batch_size=16

# 从一个学生开始蒸馏到8个学生 BIM
python3.6 -u smgd_attack201.py --output_dir=./output/densenet201/BIM/t_ --attack_method=pgd --step_size=-1 --ensemble_num=7 --batch_size=4
python3.6 -u smgd_attack201.py --output_dir=./output/densenet201/BIM/t_ --attack_method=pgd --step_size=-1 --ensemble_num=6 --batch_size=4
python3.6 -u smgd_attack201.py --output_dir=./output/densenet201/BIM/t_ --attack_method=pgd --step_size=-1 --ensemble_num=5 --batch_size=6
python3.6 -u smgd_attack201.py --output_dir=./output/densenet201/BIM/t_ --attack_method=pgd --step_size=-1 --ensemble_num=4 --batch_size=6
python3.6 -u smgd_attack201.py --output_dir=./output/densenet201/BIM/t_ --attack_method=pgd --step_size=-1 --ensemble_num=3 --batch_size=8
python3.6 -u smgd_attack201.py --output_dir=./output/densenet201/BIM/t_ --attack_method=pgd --step_size=-1 --ensemble_num=2 --batch_size=10
python3.6 -u smgd_attack201.py --output_dir=./output/densenet201/BIM/t_ --attack_method=pgd --step_size=-1 --ensemble_num=1 --batch_size=16

