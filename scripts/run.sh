#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Standard
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-7:59:00
#SBATCH --gres=gpu:4

export CUDA_HOME=/opt/cuda-9.0.176.1/
export CUDNN_HOME=/opt/cuDNN-7.0/
export STUDENT_ID=$(whoami)
export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH
export CPATH=${CUDNN_HOME}/include:$CPATH
export PATH=${CUDA_HOME}/bin:${PATH}
export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/ibm_prize_winners/data/
mkdir -p /disk/scratch/ibm_prize_winners/models/
export ROOT_DIR=/disk/scratch/ibm_prize_winners
export DATA_DIR=/disk/scratch/ibm_prize_winners/data
export MODELS_DIR=/disk/scratch/ibm_prize_winners/models


# Activate the relevant virtual environment
source /home/${STUDENT_ID}/miniconda3/bin/activate mlp
python attacks.py 

rsync -ua --progress /disk/scratch/ibm_prize_winners/models/ /home/${STUDENT_ID}/mlpcw4/models/
