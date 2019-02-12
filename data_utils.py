import torch
import torchvision
import torchvision.transforms as transforms

def load_dataset(dataset):

    dataset.lower()

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
        print("| Preparing CIFAR-10 dataset...")
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 10
    elif(dataset == 'cifar100'):
        print("| Preparing CIFAR-100 dataset...")
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 100

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=500,
                                              shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=4)

    return trainloader, testloader


mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
}


'''
def load_CIFAR10():

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

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
	                                        download=True, transform=transform_train)

	trainloader = torch.utils.data.DataLoader(trainset, batch_size=500,
	                                          shuffle=True, num_workers=4)

	testset = torchvision.datasets.CIFAR10(root='./data', train=False,
	                                       download=True, transform=transform_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=100,
	                                         shuffle=False, num_workers=4)

	return trainloader,testloader


def load_CIFAR100():

	transform_train = transforms.Compose([
	    transforms.RandomCrop(32, padding=4),
	    transforms.RandomHorizontalFlip(),
	    transforms.ToTensor(),
	    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
	])

	transform_test = transforms.Compose([
	    transforms.ToTensor(),
	    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
	])

	trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
	                                        download=True, transform=transform_train)

	trainloader = torch.utils.data.DataLoader(trainset, batch_size=500,
	                                          shuffle=True, num_workers=4)

	testset = torchvision.datasets.CIFAR100(root='./data', train=False,
	                                       download=True, transform=transform_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=100,
	                                         shuffle=False, num_workers=4)

	return trainloader,testloader
'''