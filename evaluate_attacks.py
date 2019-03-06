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
from utils.experiment_builder import ExperimentBuilder
from utils.storage_utils import dict_load
from utils.attacks import FGSMAttack,LinfPGDAttack
from utils.utils import load_net
from utils.train import adv_train

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

models = ['resnet56'] # 'densenet121']
network_names = ['cifar10', 'cifar100', 'cifar100_to_cifar10']
robust_to = ["", ]# "_fgsm", "_pgd"]
attacks = ['fgsm', 'pgd']

if torch.cuda.is_available():  # checks whether a cuda gpu is available and whether the gpu flag is True
    device = torch.device('cuda')  # sets device to be cuda
    print("use GPU")
else:
    print("use CPU")
    device = torch.device('cpu')  # sets the device to be CPU


def run_adv_evaluation(net,adversary,x,y):
        net.eval()  # sets the system to validation mode

        validaton_stat = {'clean_acc':0, 'clean_loss': 0, 'adv_acc':0, 'adv_loss': 0 }
        if len(y.shape) > 1:
            y = np.argmax(y, axis=1)  # convert one hot encoded labels to single integer labels
        if type(x) is np.ndarray:
            x, y = torch.Tensor(x).float().to(device=device), torch.Tensor(y).long().to(
            device=device)  # convert data to pytorch tensors and send to the computation device
        x = x.to(device)
        y = y.to(device)


        out = net.model(x)
        loss = F.cross_entropy(input=out, target=y)
        _,predicted = torch.max(out.data, 1)  
        accuracy = np.mean(list(predicted.eq(y.data).cpu()))
        validaton_stat['clean_acc']  = accuracy
        validaton_stat['clean_loss'] = loss
        
        # Prevent label leaking, by using most probable state
        y_pred  = pred_batch(x,net.model)

        # Create corresponding adversarial examples for training 

        # adversary = net.attacker(epsilon = 0.125)
        x_adv = adv_train(x,y_pred, net.model,nn.CrossEntropyLoss(),adversary)
        x_adv_var = to_var(x_adv)
        out = net.model(x_adv_var)
        _,predicted = torch.max(out.data, 1)  
        adv_acc = np.mean(list(predicted.eq(y.data).cpu()))
        loss_adv =  F.cross_entropy(out, y.data)

        validaton_stat['adv_acc']  = adv_acc
        validaton_stat['adv_loss'] = loss_adv

        loss = (loss + loss_adv) / 2   
        accuracy =  (accuracy + adv_acc)/2
        return loss.data.detach().cpu().numpy(), accuracy, validaton_stat

for network in network_names:
    dataset_name = {'cifar10': 'cifar10', 'cifar100': 'cifar100', 'cifar100_to_cifar10': 'cifar10'}[network]
    logging.info('\nLoading dataset: %s' %dataset_name)
    num_output_classes, train_data,val_data,test_data = getDataProviders(dataset_name=dataset_name,
                                                          rng = rng, batch_size = batch_size)
    for model in models:
        for robust in robust_to:
            experiment_name = 'attack_%s_%s%s' % (model, network, robust)
            logging.info('Experiment name: %s' %experiment_name)

            model_path =os.path.join(MODELS_DIR, "%s_%s%s/saved_models/train_model_best" % (model, network, robust))
            logging.info('Loading model from %s' % (model_path))
            net = load_net(model, model_path, num_output_classes)
            
            # Attack FGSM
            fgsmAttack = FGSMAttack(net, epsilon=0.25)
            run_adv_evaluation(net,adversary,x,y)
