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
from utils.helper_functions import load_net,freeze_layers_resnet,freeze_layers_densenet

# DATA_DIR=os.environ['DATA_DIR']
# MODELS_DIR=os.environ['MODELS_DIR']

logging.basicConfig(format='%(message)s',level=logging.INFO)

args,device = get_args()  # get arguments from command line
logging.info("Will train for %d", args.num_epochs)

rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
torch.manual_seed(seed=args.seed) # sets pytorch's seed

experiments =  {
    

        'transfer_all_layers_densenet121_nat_nat':     ('densenet121_cifar100',      'densenet121', 'nat', False, -1),
        'transfer_all_layers_densenet121_nat_fgsm':    ('densenet121_cifar100',      'densenet121', 'fgsm', True, -1),
        'transfer_all_layers_densenet121_fgsm_nat':    ('densenet121_cifar100_fgsm', 'densenet121', 'nat', False, -1),
        'transfer_all_layers_densenet121_fgsm_fgsm':   ('densenet121_cifar100_fgsm', 'densenet121', 'fgsm', True, -1), 
        'transfer_all_layers_densenet121_nat_pgd':   ('densenet121_cifar100',     'densenet121', 'pgd', True, -1),
        'transfer_all_layers_densenet121_pgd_nat':   ('densenet121_cifar100_pgd', 'densenet121', 'nat', False, -1),
        'transfer_all_layers_densenet121_pgd_pgd':   ('densenet121_cifar100_pgd', 'densenet121', 'pgd', True, -1), 

    

        'transfer_all_layers_resnet56_nat_nat':        ('resnet56_cifar100',      'resnet56', 'nat', False, -1),
        'transfer_all_layers_resnet56_fgsm_fgsm':      ('resnet56_cifar100_fgsm', 'resnet56', 'fgsm', True, -1), 
        'transfer_all_layers_resnet56_fgsm_nat':       ('resnet56_cifar100_fgsm', 'resnet56', 'nat', False, -1),
        'transfer_all_layers_resnet56_nat_fgsm':       ('resnet56_cifar100',      'resnet56', 'fgsm', True, -1),
        'transfer_all_layers_resnet56_pgd_pgd':      ('resnet56_cifar100_pgd',  'resnet56', 'pgd', True, -1), 
        'transfer_all_layers_resnet56_pgd_nat':      ('resnet56_cifar100_pgd', 'resnet56', 'nat', False, -1),
        'transfer_all_layers_resnet56_nat_pgd':      ('resnet56_cifar100',     'resnet56', 'pgd', True, -1),
    }

experiment, model, adversary, adv_train, unfrozen_layers = experiments[args.experiment_name]

num_output_classes, train_data,val_data,test_data = getDataProviders(dataset_name='cifar10', rng = rng, batch_size = args.batch_size)
num_original_classes = 100


model_path = './experiments_results/%s/saved_models/train_model_best_readable' % experiment
logging.info('Loading model from %s' % model_path)

net = load_net(model, model_path, num_original_classes)


if model == "resnet56":
    net= freeze_layers_resnet(net=net,number_of_out_classes=num_output_classes,number_of_layers=unfrozen_layers)
elif model == "densenet121":
    net= freeze_layers_densenet(net=net,number_of_out_classes=num_output_classes,number_of_layers=unfrozen_layers)
else:
    raise AssertionError('Model must be either resnet or densenet121')

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay_coefficient)
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.1)
conv_experiment = ExperimentBuilder(network_model=net,
                                    experiment_name=args.experiment_name+'_step_%d_gamma_%.1f' % (args.step_size, args.gamma),
                                    num_epochs=args.num_epochs,
                                    adv_train=adv_train,
                                    adversary=adversary,
                                    device=device,
                                    use_gpu = args.use_gpu,
                                    weight_decay_coefficient=args.weight_decay_coefficient,
                                    continue_from_epoch=args.continue_from_epoch,
                                    train_data=train_data, val_data=val_data,
                                    test_data=test_data,optimizer=optimizer,scheduler=scheduler)  # build an experiment object
                                    
experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics
