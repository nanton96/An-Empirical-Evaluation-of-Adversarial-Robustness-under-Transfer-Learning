import torchvision
from torchvision import transforms
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from utils.arg_extractor import get_args
from utils.experiment_builder import ExperimentBuilder
from utils.data_utils import getDataProviders
#from model_architectures import ConvolutionalNetwork


args = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
torch.manual_seed(seed=args.seed) # sets pytorch's seed


classes, train_data,val_data,test_data = getDataProviders(dataset_name=args.dataset_name, rng = rng, batch_size = args.batch_size)

if args.model=='resnet50':
    from utils.resnets import ResNet,BasicBlock
    # Resnet50 architecture
    net=ResNet(BasicBlock, [3, 4, 6, 3],num_classes = classes)
elif args.model=='densenet121':
    # Densetnet121 architecture
    from utils.densenets import DenseNet, Bottleneck
    net=DenseNet(Bottleneck, [6,12,24,16], growth_rate=32,num_classes = classes)
elif args.model=='resnet56':
    from utils.resnets_cifar_adapted import ResNet,BasicBlock
    net = ResNet(BasicBlock, [9, 9, 9],num_classes= classes)
else:
    raise ValueError("Model Architecture: " + args.model + " not supported")

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay_coefficient)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,milestones=[100,150],gamma=0.1)

conv_experiment = ExperimentBuilder(network_model=net,
                                    experiment_name=args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    weight_decay_coefficient=args.weight_decay_coefficient,
                                    gpu_id=args.gpu_id, use_gpu=args.use_gpu,
                                    adv_train = args.adv_train,
                                    continue_from_epoch=args.continue_from_epoch,
                                    train_data=train_data, val_data=val_data,
                                    test_data=test_data,optimizer=optimizer,scheduler=scheduler)  # build an experiment object
                                    
experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics
