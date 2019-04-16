import copy
import numpy as np
from collections import Iterable
from scipy.stats import truncnorm
import torch.nn.functional as F

import torch
import torch.nn as nn
import operator as op
import pdb

from typing import Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from utils.helper_functions import to_var, pred_batch

# --- White-box attacks ---


class FGSMAttack(object):
    def __init__(self, model=None, epsilon=None):
        """
        One step fast gradient sign method
        """
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = F.cross_entropy
        self.name = "fgsm"
    

    def perturb(self, X_nat, y, epsilons=None):

        """
        Given examples (X_nat, y), returns their adversarial
        counterparts with an attack length of epsilon.
        """
        
        # Providing epsilons in batch
        if epsilons is not None:
            self.epsilon = epsilons

        X = np.copy(X_nat)

        X_var = to_var(torch.from_numpy(X), requires_grad=True)
        y_var = to_var(torch.LongTensor(y))
        
        scores = self.model(X_var)
        loss = self.loss_fn(scores, y_var)
        loss.backward()
        grad_sign = X_var.grad.data.cpu().sign().numpy()

        X += self.epsilon * grad_sign
        # Changed from X = np.clip(X, 0, 1) 
        X = np.clip(X, -1, 1)

        return X

class LinfPGDAttack(object):
    # Change the random_start to False, as Tsipras indicated 
    def __init__(self, model=None, epsilon=0.125, k=7, a=0.03, random_start=False):
        """
        Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point.
        https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
        """
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.name = "pgd"
        self.a = a
        self.rand = random_start
        self.loss_fn = nn.CrossEntropyLoss()

    def perturb(self, X_nat, y):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        # Tsipras value 
        self.a = self.epsilon / 4

        if self.rand:
            X = X_nat + np.random.uniform(-self.epsilon, self.epsilon,
                X_nat.shape).astype('float32')
        else:
            X = np.copy(X_nat)

        for i in range(self.k):
            X_var = to_var(torch.from_numpy(X), requires_grad=True)
            y_var = to_var(torch.LongTensor(y))

            
            scores = self.model(X_var)
            loss = self.loss_fn(scores, y_var)
            loss.backward()
            grad = X_var.grad.data.cpu().numpy()

            X += self.a * np.sign(grad)

            X = np.clip(X, X_nat - self.epsilon, X_nat + self.epsilon)
            # WE changed to [-1,1] X = np.clip(X, 0, 1) 
            X = np.clip(X, -1, 1) # ensure valid pixel range

        return X

def black_box_attack(source_net,target_networks,adversary,loader,num_output_classes,device):

    accs = {target_name:[] for target_name in target_networks.keys()}   
    adversary.model = source_net 
    for x,y in loader:
        source_net.eval()
        if len(y.shape) > 1:
            y = np.argmax(y, axis=1)  # convert one hot encoded labels to single integer labels
        if type(x) is np.ndarray:
            x, y = torch.Tensor(x).float().to(device=device), torch.Tensor(y).long().to(
            device=device)  # convert data to pytorch tensors and send to the computation device
        x = x.to(device)
        y = y.to(device)       

        # Prevent label leaking        
        y_pred  = pred_batch(x,source_net)

        # Create corresponding adversarial examples for training
        if(torch.cuda.is_available()):
            x = x.cpu()
        
        x_adv = adversary.perturb(x.numpy(), y_pred)  
        x_adv = torch.from_numpy(x_adv)

        if torch.cuda.is_available():
            x_adv = x_adv.cuda()

        for target_name,target_net in target_networks.items():
            with torch.no_grad():
                target_net.eval()
                out = target_net(x_adv)
                _,predicted = torch.max(out.data, 1)  
                accuracy = np.mean(list(predicted.eq(y.data).cpu()))  # compute accuracy
                
                accs[target_name] += [accuracy]
            
    results = {target_name+'_attacked_by_' + adversary.name +'_acc':np.mean(accs[target_name]) for target_name in accs.keys()}
    return results

def attack_over_test_data(model, adversary, param, loader, device,oracle=None):
    
    accs = []
    for x,y in loader:

        model.eval()
        if len(y.shape) > 1:
            y = np.argmax(y, axis=1)  # convert one hot encoded labels to single integer labels
        if type(x) is np.ndarray:
            x, y = torch.Tensor(x).float().to(device=device), torch.Tensor(y).long().to(
            device=device)  # convert data to pytorch tensors and send to the computation device
        x = x.to(device)
        y = y.to(device)       

        out = model(x)
       
        y_pred  = pred_batch(x,model)

        # Create corresponding adversarial examples for training
        if(torch.cuda.is_available()):
            x = x.cpu()
        x_adv = adversary.perturb(x.numpy(), y_pred)  
        x_adv = torch.from_numpy(x_adv)
        if torch.cuda.is_available():
            x_adv = x_adv.cuda()
        out = model(x_adv)
        _,predicted = torch.max(out.data, 1)  
        accuracy = np.mean(list(predicted.eq(y.data).cpu()))  # compute accuracy
        accs += [accuracy]

    acc = np.mean(accs)
    print(adversary.name, "accuracy on adversarial data",acc * 100)
    return acc

def truncated_normal(mean=0.0, stddev=1.0, m=1):
    
    '''
    The generated values follow a normal distribution with specified 
    mean and standard deviation, except that values whose magnitude is 
    more than 2 standard deviations from the mean are dropped and 
    re-picked. Returns a vector of length m
    '''
    samples = []
    for i in range(m):
        while True:
            sample = np.random.normal(mean, stddev)
            if np.abs(sample) <= 2 * stddev:
                break
        samples.append(sample)
    assert len(samples) == m, "something wrong"
    if m == 1:
        return samples[0]
    else:
        return np.array(samples)

def test(model, loader,device, blackbox=False, hold_out_size=None):
    """
    Check model accuracy on model based on loader (train or test)
    """
    accs = []

    for x,y in loader:

        model.eval()
        if len(y.shape) > 1:
            y = np.argmax(y, axis=1)  # convert one hot encoded labels to single integer labels
        if type(x) is np.ndarray:
            x, y = torch.Tensor(x).float().to(device=device), torch.Tensor(y).long().to(
            device=device)  # convert data to pytorch tensors and send to the computation device
        x = x.to(device)
        y = y.to(device)
        
        out = model.forward(x)  # forward the data in the model
        
        _, predicted = torch.max(out.data, 1)  # get argmax of predictions
        
        accuracy = np.mean(list(predicted.eq(y.data).cpu()))  # compute accuracy
        accs += [accuracy]
    acc = np.mean(accs)
    print("Accuracy on clean data",acc * 100)
    return acc

