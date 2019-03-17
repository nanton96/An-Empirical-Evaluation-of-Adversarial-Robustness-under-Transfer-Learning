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

model_path =os.path.join("", "experiments_results/%s_%s/saved_models/train_model_best_readable" % (args.model, args.source_net))
logging.info('Loading %s model from %s' % (args.source_net, model_path))

net = load_net(args.model, model_path, num_original_classes)



if args.model == "resnet56":
    net= freeze_layers_resnet(net=net,number_of_out_classes=num_output_classes,number_of_layers=args.unfrozen_layers)
elif args.model == "densenet121":
    net= freeze_layers_densenet(net=net,number_of_out_classes=num_output_classes,number_of_layers=args.unfrozen_layers)
else:
    raise AssertionError

#experiment_name = 'transfer_%s_%s_to_%s_lr_%.5f_%s' % (args.model, args.source_net, args.dataset_name, args.lr, str(args.unfrozen_layers)+'_layers')
#logging.info('Experiment name: %s' %experiment_name)

# for name,param in net.named_parameters():
#     if param.requires_grad == True:
#         logging.info("REQUIRES GRAD: %s" % name)

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay_coefficient)
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5)
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
