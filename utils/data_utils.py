import logging
import torch
import torchvision
import torchvision.transforms as transforms
import utils.data_providers as data_providers

mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
}

def load_dataset(dataset, datadir):
    logging.info("Loading datasets")
    dataset.lower()
	# The output of torchvision datasets are PILImage images of range [0, 1].
	# We transform them to Tensors of normalized range [-1, 1].
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[dataset], std[dataset]),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean[dataset], std[dataset]),
    ])

    if(dataset == 'cifar10'):
        logging.info("| Preparing CIFAR-10 dataset...")
        trainset = torchvision.datasets.CIFAR10(root=datadir, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=datadir, train=False, download=True, transform=transform_test)
        num_classes = 10
    elif(dataset == 'cifar100'):
        logging.info("| Preparing CIFAR-100 dataset...")
        trainset = torchvision.datasets.CIFAR100(root=datadir, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=datadir, train=False, download=True, transform=transform_test)
        num_classes = 100

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=4)

    logging.info("Train and test datasets were loaded")
    return trainloader, testloader

def getDataProviders(dataset_name,rng,batch_size):
    if dataset_name == 'emnist':
        train_data = data_providers.EMNISTDataProvider('train', batch_size=batch_size,
                                                    rng=rng, flatten=False)  # initialize our rngs using the argument set seed
        val_data = data_providers.EMNISTDataProvider('valid', batch_size=batch_size,
                                                    rng=rng, flatten=False)  # initialize our rngs using the argument set seed
        test_data = data_providers.EMNISTDataProvider('test', batch_size=batch_size,
                                                    rng=rng, flatten=False)  # initialize our rngs using the argument set seed
        num_output_classes = train_data.num_classes

    elif dataset_name == 'cifar10':
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

        trainset = data_providers.CIFAR10(root='data', set_name='train', download=True, transform=transform_train)
        train_data = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        valset = data_providers.CIFAR10(root='data', set_name='val', download=True, transform=transform_test)
        val_data = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)

        testset = data_providers.CIFAR10(root='data', set_name='test', download=True, transform=transform_test)
        test_data = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        num_output_classes = 10

    elif dataset_name == 'cifar100':
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

        trainset = data_providers.CIFAR100(root='data', set_name='train', download=True, transform=transform_train)
        train_data = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        valset = data_providers.CIFAR100(root='data', set_name='val', download=True, transform=transform_test)
        val_data = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)

        testset = data_providers.CIFAR100(root='data', set_name='test', download=True, transform=transform_test)
        test_data = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


        num_output_classes = 100

    else:

        raise ValueError("Dataset "+ dataset_name +" name not supported")

    return num_output_classes,train_data,val_data,test_data