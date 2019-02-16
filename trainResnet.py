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
from data_utils import load_dataset
import json
# saw how to set some settings from: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--ep', default = 200, type=int, help = 'total epochs')
parser.add_argument('--loc',default=False,type=bool, help = 'Stands for local. Triggers progress_bar if not running on MLP cluster')
#parser.add_argument('--modelPath', default ='models/CIFAR10.pwf')
parser.add_argument('--dataset', default = 'cifar10', type=str, help='cifar10/cifar100')

args = parser.parse_args()

if args.loc:
    from utils import progress_bar

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
best_acc = 0  # best test accuracy
start_epoch = 1  # start from epoch 0 or last checkpoint epoch

DATA_DIR = os.environ['DATA_DIR']
MODELS_DIR = os.environ['MODELS_DIR']
checkpoint_dir = os.path.join(MODELS_DIR, "ResNet_" + args.dataset)

def train(net,trainloader,criterion,optimizer):
    # return (1s,3)
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
        # logging.info('Epoch: %d, Batch: %d Train Accuracy %.3f',epoch, batch_idx,correct/total)
        
        if args.loc:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return (train_loss, 100*correct/total)

def test(net, epoch, testloader, criterion):
    # return 2
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
            # logging.info('Batch: %d Test Accuracy %.3f',batch_idx,correct/total)
            
            if args.loc:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    
    # Save checkpoint.
    acc = 100. * correct / total
    
    if acc > best_acc:
        logging.info('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(checkpoint_dir, "ResNet_" + args.dataset + "_Best.pwf"))
        best_acc = acc
    return acc

def adv_train(net,trainloader,criterion,optimizer):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    alpha = 0.5
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        
        inputs, targets = inputs.to(device), targets.to(device)
        inputs.requires_grad = True
        optimizer.zero_grad()
        outputs = net(inputs)
        
        # Use alpha to control loss from clean and perturbed examples
        loss = criterion(outputs, targets)
        loss*=alpha
        loss.backward()
        
        # Add perturbation to inputs and send them through the network again
        
        inputs_perturbed = inputs + inputs.grad.data
        outputs = net(inputs_perturbed)
        loss = criterion(outputs, targets)
        loss *= (1-alpha)
        loss.backward()
        
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # logging.info('Epoch: %d, Batch: %d Train Accuracy %.3f',epoch, batch_idx,correct/total)
        
        if args.loc:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return (train_loss, 100 * correct / total)

def adv_test(net, epoch, testloader, criterion):
    # return 2
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
            # logging.info('Batch: %d Test Accuracy %.3f',batch_idx,correct/total)
            
            if args.loc:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    
    # Save checkpoint.
    acc = 100. * correct / total
    
    if acc > best_acc:
        logging.info('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(checkpoint_dir, "ResNet_" + args.dataset + "_Best.pwf"))
        best_acc = acc
    return acc

def train_net(net,adv=False):
   
    #Define a Loss function and optimize
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    # Changed the optimizer to exponential
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer,gamma=0.9)
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    
    net = net.to(device)
    
    if(adv):
        filename = 'train_stats.csv'
    else:
        filename = 'adv_train_stats.csv'
    if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
    
    stats={'epoch':[], 'train_acc':[], 'test_acc':[]}
    
    # print(args.ep)
    for epoch in range(start_epoch, start_epoch + args.ep):
        if(adv == False):
            train_loss, train_acc = train(net,trainloader,criterion,optimizer)
            test_acc = test(net,epoch,testloader,criterion)
        else:
            train_loss, train_acc = adv_train(net, trainloader, criterion, optimizer)
            test_acc = test(net, epoch, testloader, criterion)
        logging.info('Epoch:{:d} Loss: {:.4f} Acc: {:.1f} Test Acc: {:.1f}'.format(epoch, train_loss, train_acc, test_acc))
        stats['epoch'].append(epoch)
        stats['train_acc'].append(train_acc)
        stats['test_acc'].append(test_acc)
        # print(stats)
        with open(os.path.join(checkpoint_dir, filename), 'w+') as fp:
            json.dump(stats, fp)
        scheduler.step()
    return net
    


# ---------------- LOADING DATASETS ----------------------
trainloader, testloader = load_dataset(args.dataset, DATA_DIR)

# ---------------- LOADING ARCHITECTURE ------------------

initial_net = resnet50(pretrained=False)
trained_network = train_net(initial_net)