from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import logging
from resnets import resnet50


# In[152]:


######################################################################
# Loading the data
# ------------------
DATA_DIR = os.environ['DATA_DIR']

# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

transform = transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
        transforms.Resize(128),
#         transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))]) 
# transform from https://github.com/meliketoy/wide-resnet.pytorch/blob/master/config.py 

trainset = torchvision.datasets.CIFAR100(root=DATA_DIR, train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR100(root=DATA_DIR, train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[153]:


######################################################################
# Training the model
# ------------------
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    total_trainset = len(trainloader.dataset)

    for epoch in range(num_epochs):
        logging.info('Epoch: %d/%d' %(epoch, num_epochs));

        scheduler.step()
        model.train()  # Set model to training mode

        train_loss = 0
        correct = 0
        total = 0

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad() # zero the parameter gradients
            outputs = model(inputs) # predictions
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / total_trainset
        epoch_acc = running_corrects.double() / total_trainset

        logging.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))

        # deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logging.info('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# In[158]:


model = resnet50(pretrained=False)
model = nn.DataParallel(model) 
model.load_state_dict(torch.load("models/ResNet179.pwf", map_location=lambda storage, loc: storage))


# In[161]:


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrained model and reset final fully connected layer.
#

# model = models.resnet18(pretrained=True)



# Freeze model weights
for param in model.module.module.parameters():
    param.requires_grad = False
    
num_ftrs = model.module.module.fc.in_features
model.module.module.fc = nn.Linear(num_ftrs, 100)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^

model = train_model(model, criterion, optimizer, exp_lr_scheduler,
                       num_epochs=25)

######################################################################

