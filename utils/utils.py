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
        net=DenseNet(Bottleneck, [6,12,24,16], growth_rate=32,num_classes = num_original_classes)
    else:
        raise ValueError("Model Architecture: " + model + " not supported")

    if torch.cuda.is_available():
        model_dict = torch.load(model_path)
    else:
        model_dict = torch.load(model_path, map_location='cpu')
    # net = nn.DataParallel(module=net)
    net.load_state_dict(state_dict=model_dict['network'])
    return net

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
        out = model(to_var(x))
    y_pred = np.argmax(out.data.cpu().numpy(), axis=1)
    return torch.from_numpy(y_pred)


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
    
def batch_indices(batch_nb, data_length, batch_size):
    """
    This helper function computes a batch start and end index
    :param batch_nb: the batch number
    :param data_length: the total length of the data being parsed by batches
    :param batch_size: the number of inputs in each batch
    :return: pair of (start, end) indices
    """
    # Batch start and end index
    start = int(batch_nb * batch_size)
    end = int((batch_nb + 1) * batch_size)

    # When there are not enough inputs left, we reuse some to complete the
    # batch
    if end > data_length:
        shift = end - data_length
        start -= shift
        end -= shift

    return start, end


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

        x_adv = adversary.perturb(x.numpy(), y_pred)  
        x_adv = torch.from_numpy(x_adv)
        out = model(x_adv)
        _,predicted = torch.max(out.data, 1)  
        accuracy = np.mean(list(predicted.eq(y.data).cpu()))  # compute accuracy
        accs += [accuracy]

    acc = np.mean(accs)
    print(adversary.name,"accuracy on adversarial data",acc * 100)
    return acc

def freeze_layers_resnet(net,number_of_layers,number_of_out_classes):
    for param in net.parameters():
        param.requires_grad = False
    for i in range(9-number_of_layers,9):
        for param in net.layer3[i].parameters():
            param.requires_grad = True
    num_ftrs = net.linear.in_features
    net.linear = nn.Linear(num_ftrs, number_of_out_classes)
    return net
    
