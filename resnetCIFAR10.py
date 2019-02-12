from __future__ import print_function
import torch
import torchvision
import torchvision.transforms as transforms
from resnets import resnet50
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from FGSM import attack_network
import torch.nn.functional as F
# from utils import progress_bar
import os
import argparse
import logging

# saw how to set some settings from: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py

logging.basicConfig(filename='log_file.log', level=logging.INFO)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.12, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--ep', default = 200, type=int, help = 'total epochs')
parser.add_argument('--modelPath', default ='models/CIFAR10.pwf')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=4)

logging.info("Both datasets were downloaded")
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
net = resnet50(pretrained=False)
# net = nn.DataParallel(net, device_ids=None)


def train(epoch,trainloader):
    logging.info('Epoch: %d',epoch);
    
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        scheduler.step()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        logging.info('Batch: %d Train Accuracy %.3f',batch_idx,correct/total)
        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch,testloader):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            logging.info('Batch: %d Test Accuracy %.3f',batch_idx,correct/total)
            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc
    attack()

def attack():
    resnet = resnet50(pretrained=False)
    resnet = nn.DataParallel(resnet)
    resnet.load_state_dict(torch.load("models/ResNet179.pwf", map_location=lambda storage, loc: storage))
    resnet.eval()
    attack_network(resnet)


#Get our network Architecture

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

#Define a Loss function and optimize

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=50,gamma=0.1)

# Train the network


for epoch in range(start_epoch, start_epoch + args.ep):
    train(epoch,trainloader)
    if epoch > 95:
        torch.save(net.state_dict(), "models/ResNet{0:03d}.pwf".format(epoch))


test(epoch,testloader)

# PATH = args.modelPath
