import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import pickle
from torch.utils.data import sampler

def load_net(model, model_path, num_original_classes):
    if model=='resnet56':
        from utils.resnets_cifar_adapted import ResNet,BasicBlock
        net = ResNet(BasicBlock, [9, 9, 9],num_classes= num_original_classes)
    elif model=='densenet121':
        from utils.densenets import DenseNet, Bottleneck
        net = DenseNet(Bottleneck, [6,12,24,16], growth_rate=12,num_classes = num_original_classes)

        # net=DenseNet(Bottleneck, [6,12,24,16], growth_rate=32,num_classes = num_original_classes)
    else:
        raise ValueError("Model Architecture: " + model + " not supported")

    if torch.cuda.is_available():
        model_dict = torch.load(model_path)
    else:
        model_dict = torch.load(model_path, map_location='cpu')
    # net = nn.DataParallel(module=net)
    net.load_state_dict(state_dict=model_dict['network'])
    return net

# --- PyTorch helpers ---

def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    if volatile:
        with torch.no_grad():
            x_1 = Variable(x, requires_grad=requires_grad)
    else:
        x_1 = Variable(x, requires_grad=requires_grad)
    return x_1


def pred_batch(x, model):
    """
    batch prediction helper
    """
    with torch.no_grad():
        # x = to_var(x)
        out = model(x)
    y_pred = np.argmax(out.data.cpu().numpy(), axis=1)
    return torch.from_numpy(y_pred)


def freeze_layers_resnet(net,number_of_layers,number_of_out_classes):
    for param in net.parameters():
        param.requires_grad = False
    for i in range(9-number_of_layers,9):
        for param in net.layer3[i].parameters():
            param.requires_grad = True
    num_ftrs = net.linear.in_features
    net.linear = nn.Linear(num_ftrs, number_of_out_classes)
    return net

def freeze_layers_densenet(net,number_of_layers,number_of_out_classes):
    for param in net.parameters():
        param.requires_grad = False
    for i in range(15-number_of_layers,15):
        for param in net.dense4[i].parameters():       
            param.requires_grad = True
    num_ftrs = net.linear.in_features
    net.linear = nn.Linear(num_ftrs, number_of_out_classes)
    return net
    
