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
from utils.utils import load_net, test, attack_over_test_data
from utils.train import adv_train


DATA_DIR=os.environ['DATA_DIR']
MODELS_DIR=os.environ['MODELS_DIR']
logging.basicConfig(format='%(message)s',level=logging.INFO)

batch_size = 100

rng = np.random.RandomState(seed=0)  # set the seeds for the experiment
torch.manual_seed(seed=0) # sets pytorch's seed
# load data_set (only need test set...)

attacks = [FGSMAttack,LinfPGDAttack] 

if torch.cuda.is_available():  # checks whether a cuda gpu is available and whether the gpu flag is True
    device = torch.device('cuda')  # sets device to be cuda
    print("use GPU")
else:
    print("use CPU")
    device = torch.device('cpu')  # sets the device to be CPU

trained_networks =  {
                    'resnet56_cifar10': 'cifar10',
<<<<<<< HEAD
                    'resnet56_cifar100': 'cifar100', 
=======
                    'resnet56_cifar100': 'cifar100',
>>>>>>> 3f91e674f8f310ba816670119ee752d1e6d5f0d0
                    'resnet56_cifar100_to_cifar10': 'cifar10'
                    # 'resnet56_cifar10_fgsm_1gpu_100': 'cifar10'
                    ### Add more
                    }
results = {}

for trained_network, dataset_name, in trained_networks.items():
    model = trained_network.split('_')[0]
    logging.info('\nLoading dataset: %s' %dataset_name)
    num_output_classes, train_data,val_data,test_data = getDataProviders(dataset_name=dataset_name,rng = rng, batch_size = batch_size)
    experiment_name = 'attack_%s' % (trained_network)
    logging.info('Experiment name: %s' %experiment_name)

    model_path =os.path.join("", "experiments_results/%s/saved_models/train_model_best_readable" % (trained_network))
    logging.info('Loading model from %s' % (model_path))
    net = load_net(model, model_path, num_output_classes)
    net.to(device)
    acc = test(net,test_data,device)
    results[trained_network+"_clean"] = acc
    # Attack FGSM
    for attack in attacks:
        adversary = attack(epsilon = 0.3)
        adversary.model = net
        
        acc = attack_over_test_data(model=net,device=device ,adversary=adversary, param=None, loader=test_data, oracle=None)
        results[trained_network+"_"+adversary.name] = acc

with open('data.json', 'w') as outfile:
    json.dump(results, outfile)        

