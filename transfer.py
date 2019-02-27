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

DATA_DIR='../data'
MODELS_DIR='experiments_results'

logging.basicConfig(format='%(message)s',level=logging.INFO)

args = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
torch.manual_seed(seed=args.seed) # sets pytorch's seed


num_output_classes, train_data,val_data,test_data = getDataProviders(dataset_name=args.dataset_name, rng = rng, batch_size = args.batch_size)
num_original_classes = 10 if args.source_net == 'cifar10' else 100

model_path =os.path.join(MODELS_DIR, "%s_%s/saved_models/train_model_best" % (args.model, args.source_net))
logging.info('Loading %s model from %s' % (args.source_net, model_path))

if args.model=='resnet50':
    from utils.resnets import ResNet,BasicBlock
    net=ResNet(BasicBlock, [3, 4, 6, 3],num_classes = num_original_classes)
    model_dict = dict_load(model_path, parallel=False)
elif args.model=='resnet56':
    from utils.resnets_cifar_adapted import ResNet,BasicBlock
    net = ResNet(BasicBlock, [9, 9, 9],num_classes= num_original_classes)
    model_dict = dict_load(model_path, parallel=False)
elif args.model=='densenet121':
    from utils.densenets import DenseNet, Bottleneck
    net=DenseNet(Bottleneck, [6,12,24,16], growth_rate=32,num_classes = num_original_classes)
    # net = torch.nn.DataParallel(net)
    model_dict = dict_load(model_path, parallel=True)
else:
    raise ValueError("Model Architecture: " + args.model + " not supported")


net.load_state_dict(state_dict=model_dict)

if args.feature_extraction==True:
    for param in net.parameters():
        param.requires_grad = False
    transfer = 'last_layer'
else:
    transfer = 'all_layers'

experiment_name = 'transfer_%s_%s_to_%s_lr_%.5f_%s' % (args.model, args.source_net, args.dataset_name, args.lr, transfer)
logging.info('Experiment name: %s' %experiment_name)

# if args.model=='resnet56':
num_ftrs = net.linear.in_features
net.linear = nn.Linear(num_ftrs, num_output_classes)
# elif args.model=='densenet121':
#     num_ftrs = net.module.linear.in_features
#     net.linear = nn.Linear(num_ftrs, num_output_classes)

for name,param in net.named_parameters():
    if param.requires_grad == True:
        logging.info("REQUIRES GRAD: %s" % name)

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay_coefficient)
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1)

conv_experiment = ExperimentBuilder(network_model=net,
                                    experiment_name=experiment_name,
                                    num_epochs=args.num_epochs,
                                    weight_decay_coefficient=args.weight_decay_coefficient,
                                    gpu_id=args.gpu_id, use_gpu=args.use_gpu,
                                    continue_from_epoch=args.continue_from_epoch,
                                    train_data=train_data, val_data=val_data,
                                    test_data=test_data,optimizer=optimizer,scheduler=scheduler)  # build an experiment object
                                    
experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics
