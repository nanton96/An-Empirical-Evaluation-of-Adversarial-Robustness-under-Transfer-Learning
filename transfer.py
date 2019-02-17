import torchvision
from torchvision import transforms, models
import torch
import torch.optim as optim
import torch.nn as nn
import data_providers as data_providers
import numpy as np
from arg_extractor import get_args
from experiment_builder import ExperimentBuilder
#from model_architectures import ConvolutionalNetwork
# from resnets import resnet50

args = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
torch.manual_seed(seed=args.seed) # sets pytorch's seed

DATA_DIR='../data'

if args.dataset_name == 'emnist':
    train_data = data_providers.EMNISTDataProvider('train', batch_size=args.batch_size,
                                                   rng=rng, flatten=False)  # initialize our rngs using the argument set seed
    val_data = data_providers.EMNISTDataProvider('valid', batch_size=args.batch_size,
                                                 rng=rng, flatten=False)  # initialize our rngs using the argument set seed
    test_data = data_providers.EMNISTDataProvider('test', batch_size=args.batch_size,
                                                  rng=rng, flatten=False)  # initialize our rngs using the argument set seed
    num_output_classes = train_data.num_classes

elif args.dataset_name == 'cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = data_providers.CIFAR10(root=DATA_DIR, set_name='train', download=True, transform=transform_train)
    train_data = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

    valset = data_providers.CIFAR10(root=DATA_DIR, set_name='val', download=True, transform=transform_test)
    val_data = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=2)

    testset = data_providers.CIFAR10(root=DATA_DIR, set_name='test', download=True, transform=transform_test)
    test_data = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    num_output_classes = 10

elif args.dataset_name == 'cifar100':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = data_providers.CIFAR100(root=DATA_DIR, set_name='train', download=True, transform=transform_train)
    train_data = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    valset = data_providers.CIFAR100(root=DATA_DIR, set_name='val', download=True, transform=transform_test)
    val_data = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=2)

    testset = data_providers.CIFAR100(root=DATA_DIR, set_name='test', download=True, transform=transform_test)
    test_data = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    num_output_classes = 100

if args.model=='resnet50':
    from resnets import resnet50
    net=models.resnet50(pretrained=True)
elif args.model=='densenet121':
    from densenets import DenseNet121
    net=DenseNet121()

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay_coefficient)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,milestones=[100,150],gamma=0.1)

for param in net.parameters():
    net.requires_grad = False

num_ftrs = net.fc.in_features
# net.fc.weight.requires_grad=True
net.fc = nn.Linear(num_ftrs, num_output_classes)

conv_experiment = ExperimentBuilder(network_model=net,
                                    experiment_name=args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    weight_decay_coefficient=args.weight_decay_coefficient,
                                    gpu_id=args.gpu_id, use_gpu=args.use_gpu,
                                    continue_from_epoch=args.continue_from_epoch,
                                    train_data=train_data, val_data=val_data,
                                    test_data=test_data,optimizer=optimizer,scheduler=scheduler)  # build an experiment object
                                    
experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics