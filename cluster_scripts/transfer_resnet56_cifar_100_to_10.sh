#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Interactive
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-01:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}


export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

mkdir -p ${TMP}/datasets/
export DATASET_DIR=${TMP}/datasets/
# Activate the relevant virtual environment:

export MODELS_DIR=/home/${STUDENT_ID}/mlpcw4/experiments_results
export DATA_DIR=/disk/scratch/${STUDENT_ID}/data $DATA_DIR

rsync -ua --progress /home/${STUDENT_ID}/mlpcw4/data/ /disk/scratch/${STUDENT_ID}/data

source /home/${STUDENT_ID}/miniconda3/bin/activate mlp
cd ..


python transfer.py --batch_size 100 --continue_from_epoch -1 --seed 0 \
                 --adv_train False \
                 --num_epochs 25 \
                 --adversary "fgsm" \
                 --lr 0.1 --model 'resnet56' \
                 --source_net cifar100 \
                 --experiment_name 'transfer_resnet56_cifar100_to_10' \
                 --use_gpu True --gpu_id "0" --weight_decay_coefficient 0.00005 \
                 --unfrozen_layers 5 \
                 --dataset_name "cifar10"


python transfer.py --batch_size 100 --continue_from_epoch -1 --seed 0 \
                 --adv_train False \
                 --num_epochs 25 \
                 --adversary "fgsm" \
                 --lr 0.1 --model 'densetnet121' \
                 --source_net cifar100 \
                 --experiment_name 'transfer_densetnet121_cifar100_to_10' \
                 --use_gpu True --gpu_id "0" --weight_decay_coefficient 0.00005 \
                 --unfrozen_layers 5 \
                 --dataset_name "cifar10"
