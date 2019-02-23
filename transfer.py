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

DATA_DIR='../data'
MODELS_DIR='experiments_results'

logging.basicConfig(format='%(message)s',level=logging.INFO)

args = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
torch.manual_seed(seed=args.seed) # sets pytorch's seed

experiment_name = 'transfer_%s_%s_to_%s_lr_%.5f' % (args.model, args.source_net, args.dataset_name, args.lr)
logging.info('Experiment name: %s' %experiment_name)

num_output_classes, train_data,val_data,test_data = getDataProviders(dataset_name=args.dataset_name, rng = rng, batch_size = args.batch_size)

num_original_classes = 10
if args.model=='resnet50':
    from utils.resnets import ResNet,BasicBlock
    # Resnet50 architecture
    net=ResNet(BasicBlock, [3, 4, 6, 3],num_classes = num_original_classes)
elif args.model=='densenet121':
    # Densetnet121 architecture
    from utils.densenets import DenseNet, Bottleneck
    net=DenseNet(Bottleneck, [6,12,24,16], growth_rate=32,num_classes = num_original_classes)
elif args.model=='resnet56':
    from utils.resnets_cifar_adapted import ResNet,BasicBlock
    net = ResNet(BasicBlock, [9, 9, 9],num_classes= num_original_classes)
else:
    raise ValueError("Model Architecture: " + args.model + " not supported")

model_path =os.path.join(MODELS_DIR, "%s_%s/saved_models/train_model_best" % (args.model, args.source_net))
logging.info('Loading %s model from %s' % (args.source_net, model_path))
if torch.cuda.is_available():
    model_dict = torch.load(model_path)
else:
    model_dict = torch.load(model_path, map_location='cpu')
model_dict2 = {}
for k,v in model_dict['network'].items():
    model_dict2[k[6:]] = v
# net = torch.nn.DataParallel(net)
net.load_state_dict(state_dict=model_dict2)

print(model_dict2.keys())

# state = torch.load(f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx))))
# self.load_state_dict(state_dict=state['network'])

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay_coefficient)
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1)

for param in net.parameters():
    net.requires_grad = False

num_ftrs = net.linear.in_features
# net.fc.weight.requires_grad=True
net.linear = nn.Linear(num_ftrs, num_output_classes)


conv_experiment = ExperimentBuilder(network_model=net,
                                    experiment_name=experiment_name,
                                    num_epochs=args.num_epochs,
                                    weight_decay_coefficient=args.weight_decay_coefficient,
                                    gpu_id=args.gpu_id, use_gpu=args.use_gpu,
                                    continue_from_epoch=args.continue_from_epoch,
                                    train_data=train_data, val_data=val_data,
                                    test_data=test_data,optimizer=optimizer,scheduler=scheduler)  # build an experiment object
                                    
experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics
