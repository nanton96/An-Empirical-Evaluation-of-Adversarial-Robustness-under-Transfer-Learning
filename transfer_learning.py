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
from data_utils import load_dataset

logging.basicConfig(format='%(message)s', level=logging.INFO)

# ------------------
DATA_DIR = os.environ['DATA_DIR']
MODELS_DIR = os.environ['MODELS_DIR']

# ---------------- LOADING DATASETS ----------------------
trainloader, testloader = load_dataset('cifar100', DATA_DIR)


# ---------------- SET GPU DEVICES ----------------------
gpu_id = "0,1"
if torch.cuda.is_available():  # checks whether a cuda gpu is available and whether the gpu flag is True
	if "," in gpu_id:
		device = [torch.device('cuda:{}'.format(idx)) for idx in gpu_id.split(",")]  # sets device to be cuda
	else:
		device = torch.device('cuda:{}'.format(gpu_id))  # sets device to be cuda

	os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id  # sets the main GPU to be the one at index 0 (on multi gpu machines you can choose which one you want to use by using the relevant GPU ID)
	print("use GPU")
	print("GPU ID {}".format(gpu_id))
else:
	print("use CPU")
	device = torch.device('cpu')  # sets the device to be CPU

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
			epoch, epoch_loss, epoch_acc))

		checkpoint_dir = os.path.join(MODELS_DIR, "transfer_learning")
		if not os.path.isdir(checkpoint_dir):
        		os.mkdir(checkpoint_dir)
		# deep copy the model
		if epoch_acc > best_acc:
			best_acc = epoch_acc
			best_model_wts = copy.deepcopy(model.state_dict())
			torch.save(model, os.path.join(checkpoint_dir, "ResNet_cifar10_to_100_Best.pwf"))
			
	time_elapsed = time.time() - since
	logging.info('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	logging.info('Best val Acc: {:4f}'.format(best_acc))

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model

######################################################################
# Finetuning the convnet
# ----------------------
#

# Load a pretrained model and reset final fully connected layer.

model = resnet50(pretrained=False)
mpath =os.path.join(MODELS_DIR, "ResNet_cifar10/ResNet_cifar10_Best.pwf")
logging.info(mpath) 
mdict = torch.load(os.path.join(MODELS_DIR, "ResNet_cifar10/ResNet_cifar10_Best.pwf"))
model.load_state_dict(mdict['net'])

# Freeze model weights
for param in model.parameters():
	param.requires_grad = False
	
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 100)

###############################################
############ Parallelize model ################
if type(device) is list:
	model.to(device[0])
	model = nn.DataParallel(module=model, device_ids=device)
	device = device[0]
else:
	model.to(device)  # sends the model from the cpu to the gpu



model = model.to(device)
criterion = nn.CrossEntropyLoss().to(device)  
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
