#!/bin/bash

# An array of models to loop over
declare -a models=("resnet18" "LeNet" "Inception")
declare -a num_bits=("8")
declare -a numberofcentroids=("3")

# Loop over each model
for model in "${models[@]}"
do
   for bit in "${num_bits[@]}"
   do
      for centroid in "${numberofcentroids[@]}"
      do
         python3 main.py --method all --gpu 1 --model $model --num_bits $bit --numberofcentroids $centroid --save 1 --duplicate 1
      done
   done
done
