#!/bin/bash

Model=(densenet201)

T=(20 22 24 26 28 30)

clear
for t in "${T[@]}"
do
  for model in "${Model[@]}"
  do
    python3 -u new_train_kd.py --T=$t --arch=$model
  done
done