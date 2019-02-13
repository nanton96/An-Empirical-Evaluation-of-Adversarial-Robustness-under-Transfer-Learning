from __future__ import print_function
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
path = "models/ResNet_cifar10/ResNet_cifar10_Best.pwf"

from resnets import resnet50


def attack():
    dic = torch.load(path,map_location='cpu')
    resnet = resnet50()
    resnet.load_state_dict(dic['net'])
    resnet.eval()
    print("Network was loaded, attacking")
    attack_network(resnet)
    
def attack_network(model):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    testset = datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=True, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = {'epsilon': [], 'accuracy': []}
    epsilons = [0.05, 0.1, 0.2, 0.25, 0.4]
    for epsilon in epsilons:
        check_robustness(model,device,testloader,stats,epsilon)
    
def check_robustness(model,device,test_loader,stats,epsilon=.25):
    correct = 0
    adv_examples = []
    

    print("Begging attack")
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        
        # loss w.r.t the inputs will be automatically computed
        data.requires_grad = True
        
        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        
        # If the initial prediction is wrong, dont bother attacking, just move on
        # if init_pred.item() != target.item():
        #     continue
        
        # Calculate the loss
        loss = F.nll_loss(output, target)
        
        # Zero all existing gradients
        model.zero_grad()
        
        # Calculate gradients of model in backward pass
        loss.backward()
        
        # Collect datagrad
        data_grad = data.grad.data
        
        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        
        # Re-classify the perturbed image
        output = model(perturbed_data)
        
        # Check for success
        final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        
        # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(test_loader))
    stats['epsilon'].append(epsilon)
    stats['accuracy'].append(final_acc)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))
    
    # Return the accuracy and an adversarial example
    return final_acc, adv_examples
    pass

def fgsm_attack(image, epsilon, data_grad):
    print("Applying FGSM to image")
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    # Return the perturbed image
    return perturbed_image

attack()