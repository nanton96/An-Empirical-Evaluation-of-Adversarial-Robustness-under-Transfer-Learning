import torchvision
from torchvision import transforms, models
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import logging
import os
from utils.data_utils import getDataProviders
from utils.arg_extractor import get_args
from utils.storage_utils import dict_load
from utils.attacks import FGSMAttack,LinfPGDAttack
from utils.utils import load_net, test, attack_over_test_data
from utils.train import adv_train

DATA_DIR=os.environ['DATA_DIR']
MODELS_DIR=os.environ['MODELS_DIR']
logging.basicConfig(format='%(message)s',level=logging.INFO)

batch_size = 100

rng = np.random.RandomState(seed=0)  # set the seeds for the experiment
torch.manual_seed(seed=0) # sets pytorch's seed
# load data_set (only need test set...)


models = ['resnet56'] # 'densenet121']
trained_datasets = ['cifar10','cifar100', 'cifar100_to_cifar10']
robust_to = ["", ]# "_fgsm", "_pgd"]
attacks = [FGSMAttack,LinfPGDAttack] 

if torch.cuda.is_available():  # checks whether a cuda gpu is available and whether the gpu flag is True
    device = torch.device('cuda')  # sets device to be cuda
    print("use GPU")
else:
    print("use CPU")
    device = torch.device('cpu')  # sets the device to be CPU
dataset_for_each_network =  {'cifar10': 'cifar10', 'cifar100': 'cifar100', 'cifar100_to_cifar10': 'cifar10'}

for trained_dataset in trained_datasets:
    dataset_name =dataset_for_each_network[trained_dataset]
    logging.info('\nLoading dataset: %s' %dataset_name)
    num_output_classes, train_data,val_data,test_data = getDataProviders(dataset_name=dataset_name,rng = rng, batch_size = batch_size)
    for model in models:
        for robust in robust_to:
            experiment_name = 'attack_%s_%s%s' % (model, trained_dataset, robust)
            logging.info('Experiment name: %s' %experiment_name)

            model_path =os.path.join(MODELS_DIR, "%s_%s%s/saved_models/train_model_best" % (model, trained_dataset, robust))
            logging.info('Loading model from %s' % (model_path))
            net = load_net(model, model_path, num_output_classes)
            acc = test(net,test_data,device)
            # Attack FGSM
            for attack in attacks:
                adversary = attack(epsilon = 0.125)
                adversary.model = net
                #   acc = attack_over_test_data(model=net, adversary=adversary, param=None, loader_test=test_data, oracle=None)
                



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