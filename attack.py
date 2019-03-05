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

DATA_DIR=os.environ['DATA_DIR']
MODELS_DIR=os.environ['MODELS_DIR']

logging.basicConfig(format='%(message)s',level=logging.INFO)

args = get_attack_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
torch.manual_seed(seed=args.seed) # sets pytorch's seed

# load data_set (only need test set...)
num_output_classes, train_data,val_data,test_data = getDataProviders(dataset_name=args.dataset_name, rng = rng, batch_size = args.batch_size)


model_path =os.path.join(MODELS_DIR, "%s_%s/saved_models/train_model_best" % (args.model, args.source_net))
logging.info('Loading %s model from %s' % (args.source_net, model_path))

model = args.model

def load_model_architecture(model,num_classes):
    if model=='resnet50':
        from utils.resnets import ResNet,BasicBlock
        net=ResNet(BasicBlock, [3, 4, 6, 3],num_classes = num_original_classes)
        model_dict = dict_load(model_path, parallel=False)
    elif model=='resnet56':
        from utils.resnets_cifar_adapted import ResNet,BasicBlock
        net = ResNet(BasicBlock, [9, 9, 9],num_classes= num_original_classes)
        model_dict = dict_load(model_path, parallel=False)
    elif model=='densenet121':
        from utils.densenets import DenseNet, Bottleneck
        net=DenseNet(Bottleneck, [6,12,24,16], growth_rate=32,num_classes = num_original_classes)
        # net = torch.nn.DataParallel(net)
        model_dict = dict_load(model_path, parallel=True)
    else:
        raise ValueError("Model Architecture: " + model + " not supported")
    return net,model_dict

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
