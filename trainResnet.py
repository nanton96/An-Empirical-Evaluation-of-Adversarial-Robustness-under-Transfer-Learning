from __future__ import print_function
import torch
import torchvision

from resnets import resnet50
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
import argparse
import logging
import FGSM
from data_utils import load_dataset
# saw how to set some settings from: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--ep', default = 200, type=int, help = 'total epochs')
parser.add_argument('--loc',default=False,type=bool, help = 'Stands for local. Triggers progress_bar if not running on MLP cluster')
#parser.add_argument('--modelPath', default ='models/CIFAR10.pwf')
parser.add_argument('--dt', default = 'cifar10', type=str, help='cifar10/cifar100')

args = parser.parse_args()

if args.loc:
    from utils import progress_bar

logging.basicConfig(filename= 'log_file' + 'ResNet' + args.dt + '.log', level=logging.INFO)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# ---------------- LOADING DATASETS ----------------------
trainloader, testloader = load_dataset(args.dt)
logging.info("Train and test datasets were loaded")

# ---------------- LOADING ARCHITECTURE ------------------
net = resnet50(pretrained=False)

def train(epoch,trainloader):
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
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        logging.info('Epoch: %d, Batch: %d Train Accuracy %.3f',epoch, batch_idx,correct/total)
        
        if args.loc:
        
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

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
            
            if args.loc:
                
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        
        logging.info('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        
        torch.save(state, './checkpoint/ckpt.t7')
        torch.save(net.state_dict(), "models/ResNet"+args.dt+"Best.pwf".format(epoch))
        best_acc = acc

#Get our network Architecture

if device == 'cuda':
    cudnn.benchmark = True
    net = nn.DataParallel(net, device_ids=None)

#Define a Loss function and optimize
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=50,gamma=0.1)

# Train the network

for epoch in range(start_epoch, start_epoch + args.ep):
    train(epoch,trainloader)
    #if epoch > 95:
    #    torch.save(net.state_dict(), "models/ResNet{0:03d}.pwf".format(epoch))
    test(epoch,testloader)
    scheduler.step()

