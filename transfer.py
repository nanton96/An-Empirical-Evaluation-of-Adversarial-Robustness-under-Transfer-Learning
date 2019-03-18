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
from utils.utils import load_net,freeze_layers_resnet,freeze_layers_densenet

# DATA_DIR=os.environ['DATA_DIR']
# MODELS_DIR=os.environ['MODELS_DIR']

logging.basicConfig(format='%(message)s',level=logging.INFO)

args,device = get_args()  # get arguments from command line
logging.info("Will train for", args.num_epochs)

rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
torch.manual_seed(seed=args.seed) # sets pytorch's seed


num_output_classes, train_data,val_data,test_data = getDataProviders(dataset_name=args.dataset_name, rng = rng, batch_size = args.batch_size)
num_original_classes = 10 if args.source_net == 'cifar10' else 100

if args.trained_on:
    args.source_net +='_'+args.trained_on
model_path =os.path.join("", "experiments_results/%s_%s/saved_models/train_model_best_readable" % (args.model, args.source_net))
logging.info('Loading %s model from %s' % (args.source_net, model_path))

net = load_net(args.model, model_path, num_original_classes)

if args.model == "resnet56" or  args.model == "resnet56_fgsm" or args.model == "resnet56_pgd":
    net= freeze_layers_resnet(net=net,number_of_out_classes=num_output_classes,number_of_layers=args.unfrozen_layers)
elif args.model == "densenet121" or args.model == "densenet121_fgsm" or args.model == "densenet121_pgd":
    net= freeze_layers_densenet(net=net,number_of_out_classes=num_output_classes,number_of_layers=args.unfrozen_layers)
else:
    raise AssertionError


experiments()



optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay_coefficient)
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.1)
conv_experiment = ExperimentBuilder(network_model=net,
                                    experiment_name=args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    adv_train=args.adv_train,
                                    adversary=args.adversary,
                                    device=device,
                                    use_gpu = args.use_gpu,
                                    weight_decay_coefficient=args.weight_decay_coefficient,
                                    continue_from_epoch=args.continue_from_epoch,
                                    train_data=train_data, val_data=val_data,
                                    test_data=test_data,optimizer=optimizer,scheduler=scheduler)  # build an experiment object
                                    
experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics
