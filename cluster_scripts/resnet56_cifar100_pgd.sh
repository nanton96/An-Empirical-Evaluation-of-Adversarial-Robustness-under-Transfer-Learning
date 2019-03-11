#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=LongJobs
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-39:59:59

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

rsync -ua --progress /home/${STUDENT_ID}/mlpcw4/data/ /disk/scratch/${STUDENT_ID}/data

source /home/${STUDENT_ID}/miniconda3/bin/activate mlp
cd ..
mkdir experiments_results

python train.py --batch_size 100 \
				--num_epochs 200 \
				--model resnet56 \
				--dataset_name "cifar100" \
				--adv_train True \
				--adversary "pgd" \
				--experiment_name 'resnet56_cifar100_pgd_1gpu_100b' \
				--use_gpu "True" \
				--lr 0.1 \
				--continue_from_epoch -1 \
				--seed 0 \

