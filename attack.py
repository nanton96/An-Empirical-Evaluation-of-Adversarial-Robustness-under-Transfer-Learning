import torchvision
from torchvision import transforms, models
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import logging
import os
from utils.data_utils import getDataProviders
from utils.arg_extractor import get_args
from utils.experiment_builder import ExperimentBuilder
from utils.storage_utils import dict_load
from utils.attacks import FGSMAttack,LinfPGDAttack
from utils.utils import load_net

DATA_DIR=os.environ['DATA_DIR']
MODELS_DIR=os.environ['MODELS_DIR']
logging.basicConfig(format='%(message)s',level=logging.INFO)

batch_size = 100

rng = np.random.RandomState(seed=0)  # set the seeds for the experiment
torch.manual_seed(seed=0) # sets pytorch's seed
# load data_set (only need test set...)


########## PSEUDO CODE ##########

# Iterate over models and attacks
    # LOAD MODEL
        # # load model architecture, dict
    # ATTACK 
        # # x_adv = apply attack on test_set wrt to model params
        # # forward pass the x_adv through the network to generate y'
        # # calculate accuracy and store in dictionary under key 'model_attack_acc'
# SAVE
    # dump dictionary into pickle file


# For black_box

    # LOAD source Model, defence Model

    # Attack source
        # x_adv = apply attack on test_set wrt to source model params
    # Evaluate on target
        # # forward pass the x_adv through defender network to generate y'
        # # calculate accuracy and store in dictionary under key 'model_attack_black_box(source)_acc'

models = ['resnet56',]# 'densenet121']
network_names = ['cifar10', 'cifar100', 'cifar100_to_cifar10']
adversary_attacks = ["", ]# "_fgsm", "_pgd"]

for network in network_names:
    dataset_name = {'cifar10': 'cifar10', 'cifar100': 'cifar100', 'cifar100_to_cifar10': 'cifar10'}[network]
    logging.info('\nLoading dataset: %s' %dataset_name)
    num_output_classes, train_data,val_data,test_data = getDataProviders(dataset_name=dataset_name,
                                                          rng = rng, batch_size = batch_size)
    for model in models:
        for adversary in adversary_attacks:
            experiment_name = 'attack_%s_%s%s' % (model, network, adversary)
            logging.info('Experiment name: %s' %experiment_name)

            model_path =os.path.join(MODELS_DIR, "%s_%s%s/saved_models/train_model_best" % (model, network, adversary))
            logging.info('Loading model from %s' % (model_path))
            net = load_net(model, model_path, num_output_classes)
    