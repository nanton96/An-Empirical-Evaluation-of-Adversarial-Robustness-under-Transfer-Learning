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
from utils.helper_functions import load_net
from utils.evaluation_functions import test, attack_over_test_data, FGSMAttack, LinfPGDAttack
from utils.train import adv_train
from scipy.stats import truncnorm

logging.basicConfig(format='%(message)s',level=logging.INFO)

batch_size = 100

rng = np.random.RandomState(seed=0)  # set the seeds for the experiment
torch.manual_seed(seed=0) # sets pytorch's seeds
# load data_set (only need test set...)

attacks = [FGSMAttack,lambda epsilon: LinfPGDAttack(epsilon=epsilon, k=7)] 
lower, upper,mu, sigma = 0, 0.125,0, 0.0625
distribution = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

if torch.cuda.is_available():  # checks whether a cuda gpu is available and whether the gpu flag is True
    device = torch.device('cuda')  # sets device to be cuda
    logging.info("use GPU")
else:
    logging.info("use CPU")
    device = torch.device('cpu')  # sets the device to be CPU

trained_networks =  {
                    # "transfer_densenet121_fgsm_fgsm": ('cifar10', 'densenet121'),
                    # "transfer_densenet121_fgsm_nat": ('cifar10', 'densenet121'),
                    # "transfer_densenet121_nat_fgsm": ('cifar10', 'densenet121'),
                    # "transfer_densenet121_nat_nat": ('cifar10', 'densenet121'),
                    # "transfer_densenet121_nat_pgd": ('cifar10', 'densenet121'),
                    # "transfer_densenet121_pgd_nat": ('cifar10', 'densenet121'),
                    # "transfer_densenet121_pgd_pgd": ('cifar10', 'densenet121'),
                    # "transfer_resnet56_fgsm_fgsm": ('cifar10', 'resnet56'),
                    # "transfer_resnet56_fgsm_nat": ('cifar10', 'resnet56'),
                    # "transfer_resnet56_nat_fgsm": ('cifar10', 'resnet56'),
                    # "transfer_resnet56_nat_nat": ('cifar10', 'resnet56'),
                    # "transfer_resnet56_nat_pgd": ('cifar10', 'resnet56'),
                    # "transfer_resnet56_pgd_nat": ('cifar10', 'resnet56'),
                    # "transfer_resnet56_pgd_pgd": ('cifar10', 'resnet56'),

                    # CIFAR10
                    "resnet56_cifar10_fgsm": ('cifar10', 'resnet56'),
                    # "resnet56_cifar10": ('cifar10', 'resnet56'),
                    "resnet56_cifar10_fgsm_ll": ('cifar10', 'resnet56'),
                    "resnet56_cifar10": ('cifar10', 'resnet56'),
                    # "resnet56_cifar10_pgd": ('cifar10', 'resnet56'),
                    # "densenet121_cifar10_fgsm": ('cifar10', 'densenet121'),
                    # "densenet121_cifar10": ('cifar10', 'densenet121'),
                    # "densenet121_cifar10_pgd": ('cifar10', 'densenet121'),
                    "resnet56_cifar10_fgsm_prev": ('cifar10', 'resnet56'),
                    "resnet56_cifar100_fgsm_prev": ('cifar100', 'resnet56'),
  
                    
                    # CIFAR100
                    "resnet56_cifar100_fgsm": ('cifar100', 'resnet56'),
                    "resnet56_cifar100_fgsm_ll": ('cifar100', 'resnet56'),
                    "resnet56_cifar100": ('cifar100', 'resnet56'),
                    # "resnet56_cifar100_pgd": ('cifar100', 'resnet56'),
                    # "densenet121_cifar100_fgsm": ('cifar100', 'densenet121'),
                    # "densenet121_cifar100": ('cifar100', 'densenet121'),
                    # "densenet121_cifar100_pgd": ('cifar100', 'densenet121'),


                    # "resnet56_cifar10_pgd_tsipras": ('cifar10', 'resnet56'),
                    # "resnet56_cifar10_fgsm_tsipras": ('cifar10', 'resnet56'),
                    # "densenet121_cifar10_fgsm_tsipras": ('cifar10', 'densenet121'),

                    # "transfer_all_layers_densenet121_fgsm_fgsm_step_25_gamma_0.1": ('cifar10', 'densenet121'),
                    # "transfer_all_layers_densenet121_fgsm_nat_step_25_gamma_0.1": ('cifar10', 'densenet121'),
                    # "transfer_all_layers_densenet121_fgsm_nat_step_25_gamma_0.1_2": ('cifar10', 'densenet121'),
                    # "transfer_all_layers_densenet121_nat_fgsm_step_25_gamma_0.1": ('cifar10', 'densenet121'),
                    # "transfer_all_layers_densenet121_nat_nat_step_25_gamma_0.1": ('cifar10', 'densenet121'),
                    # "transfer_all_layers_densenet121_nat_nat_step_25_gamma_0.1_2": ('cifar10', 'densenet121'),
                    # "transfer_all_layers_densenet121_nat_pgd_step_25_gamma_0.1": ('cifar10', 'densenet121'),
                    # "transfer_all_layers_densenet121_pgd_nat_step_25_gamma_0.1": ('cifar10', 'densenet121'),
                    # "transfer_all_layers_densenet121_pgd_nat_step_25_gamma_0.1_2": ('cifar10', 'densenet121'),
                    # "transfer_all_layers_densenet121_pgd_pgd_step_25_gamma_0.1": ('cifar10', 'densenet121'),
                    # "transfer_all_layers_resnet56_fgsm_fgsm_step_25_gamma_0.1": ('cifar10', 'resnet56'),
                    # "transfer_all_layers_resnet56_fgsm_fgsm_step_25_gamma_0.1_2": ('cifar10', 'resnet56'),
                    # "transfer_all_layers_resnet56_fgsm_nat_step_25_gamma_0.1": ('cifar10', 'resnet56'),
                    # "transfer_all_layers_resnet56_fgsm_nat_step_25_gamma_0.1_2": ('cifar10', 'resnet56'),
                    # "transfer_all_layers_resnet56_nat_fgsm_step_25_gamma_0.1": ('cifar10', 'resnet56'),
                    # "transfer_all_layers_resnet56_nat_fgsm_step_25_gamma_0.1_2": ('cifar10', 'resnet56'),
                    # "transfer_all_layers_resnet56_nat_nat_step_25_gamma_0.1": ('cifar10', 'resnet56'),
                    # "transfer_all_layers_resnet56_nat_pgd_step_25_gamma_0.1": ('cifar10', 'resnet56'),
                    # "transfer_all_layers_resnet56_pgd_nat_step_25_gamma_0.1": ('cifar10', 'resnet56'),
                    # "transfer_all_layers_resnet56_pgd_nat_step_25_gamma_0.1_2": ('cifar10', 'resnet56'),
                    # "transfer_all_layers_resnet56_pgd_pgd_step_25_gamma_0.1": ('cifar10', 'resnet56'),
                    # "transfer_all_layers_resnet56_nat_nat_step_20_gamma_0.4": ('cifar10', 'resnet56'),
                    }

for trained_network, (dataset_name, model) in trained_networks.items():
    results = {}
    # model = trained_network.split('_')[0]
    logging.info('\nLoading dataset: %s' %dataset_name)
    num_output_classes, train_data,val_data,test_data = getDataProviders(dataset_name=dataset_name,rng = rng, batch_size = batch_size)
    experiment_name = 'attack_%s' % (trained_network)
    logging.info('Experiment name: %s' %experiment_name)


    model_path =os.path.join('./experiments_results/', "%s/saved_models/train_model_best_readable" % (trained_network))
    logging.info('Loading model from %s' % (model_path))
    net = load_net(model, model_path, num_output_classes)
    net.to(device)
    acc = test(net,test_data,device)
    results[trained_network+"_clean"] = acc
    # Attack FGSM
    for e in [0.125/4, 0.125/2, 0.125]:
        logging.info('EPSILON %.4f' % e)
        for attack in attacks:
            adversary = attack(epsilon = e)
            # e=0.125 #e = distribution.rvs(1)[0]
            adversary.model = net
            acc = attack_over_test_data(model=net,device=device ,adversary=adversary, param=None, loader=test_data, oracle=None)
            results[trained_network+"_attacked_by_"+adversary.name+'_e_%.5f'%e] = acc

    path = './attack_results/white_box/white_box_attacks_pgd_7_TEST_%s.json' % trained_network
    with open(path, 'w') as outfile:
        json.dump(results, outfile)        

