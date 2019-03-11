import torchvision
from torchvision import transforms, models
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import logging
import os
import json
from utils.data_utils import getDataProviders
from utils.arg_extractor import get_args
from utils.storage_utils import dict_load
from utils.attacks import FGSMAttack,LinfPGDAttack
from utils.utils import load_net,black_box_attack
from utils.train import adv_train


DATA_DIR=os.environ['DATA_DIR']
MODELS_DIR=os.environ['MODELS_DIR']
logging.basicConfig(format='%(message)s',level=logging.INFO)

batch_size = 100

rng = np.random.RandomState(seed=0)  # set the seeds for the experiment
torch.manual_seed(seed=0) # sets pytorch's seed
# load data_set (only need test set...)

attacks = [FGSMAttack]#,LinfPGDAttack] 

if torch.cuda.is_available():  # checks whether a cuda gpu is available and whether the gpu flag is True
    device = torch.device('cuda')  # sets device to be cuda
    print("use GPU")
else:
    print("use CPU")
    device = torch.device('cpu')  # sets the device to be CPU

source_networks = { 'cifar10': 'resnet56_cifar10',
                    'cifar100': 'resnet56_cifar100',
}

target_networks =  {
                    'cifar10': ['resnet56_cifar10'],#,'resnet56_cifar10_fgsm','resnet56_cifar10_pgd'],
                    'cifar100': ['resnet56_cifar100']#,'resnet56_cifar100_fgsm','resnet56_cifar100_pgd']
                    }
results = []


for dataset_name,source_network in source_networks.items():
    logging.info('\nLoading dataset: %s' %dataset_name)

    num_output_classes, train_data,val_data,test_data = getDataProviders(dataset_name=dataset_name,rng = rng, batch_size = batch_size)

    model_path =os.path.join("", "experiments_results/%s/saved_models/train_model_best_readable" % (source_network))
    source_architecture = source_network.split('_')[0]
    source_net = load_net(source_architecture, model_path, num_output_classes).to(device)
    
    target_nets = {}
    for target_network in target_networks[dataset_name]:
        model_path =os.path.join("", "experiments_results/%s/saved_models/train_model_best_readable" % (target_network))
        target_architecture = target_network.split('_')[0]
        target_nets[target_network] = load_net(target_architecture, model_path, num_output_classes).to(device)
    for adversary in attacks:
        adversary = adversary(epsilon = 0.3)
        results.append(black_box_attack(source_net=source_net,target_networks=target_nets,adversary=adversary,loader=test_data,num_output_classes=num_output_classes,device=device))
        logging.info("blackbox attack for adversary: %s completed" %adversary.name)

with open('black_box_results.json', 'w') as outfile:
    json.dump(results, outfile)        
