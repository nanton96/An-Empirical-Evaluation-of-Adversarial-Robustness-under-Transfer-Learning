#!/bin/sh

export DATASET_DIR="data/"
# Activate the relevant virtual environment:
conda activate mlp
mkdir experiments_results

#python train.py --batch_size 100 --continue_from_epoch -1 --seed 0 \
#                                                    --adv_train True \
#                                                    --adversary "fgsm" \
#                                                    --lr 0.1 --model 'densenet121' \
#                                                    --num_epochs 1 --experiment_name 'cifar10_test_exp' \
#                                                    --use_gpu "False" --gpu_id "None" --weight_decay_coefficient 0.00005 \
#                                                    --dataset_name "cifar10"

 python transfer.py --batch_size 100 --continue_from_epoch -1 --seed 0 \
                     --adv_train False \
                     --adversary "fgsm" \
                     --lr 0.1 --model 'resnet56' \
                     --source_net 'cifar100' \
                     --num_epochs 25 --experiment_name 'resnet56_cifar100_test_10' \
                     --use_gpu "False" --gpu_id "0" --weight_decay_coefficient 0.00005 \
                     --unfrozen_layers 4 \
                     --dataset_name 'cifar10'

