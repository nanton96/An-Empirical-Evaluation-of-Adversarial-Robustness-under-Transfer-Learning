from __future__ import print_function
import torch
import json
import torch.nn.functional as F
import data_providers as data_providers
from torchvision import datasets, transforms
from resnets import resnet50,resnet101,resnet152
import numpy as np
import abc
import json



class attacks():

    # List of adversarial examples created
    adv_examples = []
    stats = None
    model_path = "models/"
    stats_paths = "statistics/"
    network = torch.nn.Module()
    experiment_name = None
    def __init__(self,params,model,dataset):
            self.params = params
            self.model = model
            self.dataset = dataset
        
        
    @abc.abstractmethod
    def attack(self):
        pass
    
    

    def return_dataset(self,dataset_name,batch_size=100,seed=0):
        rng = np.random.RandomState(seed=seed)
        if dataset_name == 'emnist':
            train_data = data_providers.EMNISTDataProvider('train', batch_size=batch_size,
                                                           rng=rng,
                                                           flatten=False)  # initialize our rngs using the argument set seed
            val_data = data_providers.EMNISTDataProvider('valid', batch_size=batch_size,
                                                         rng=rng,
                                                         flatten=False)  # initialize our rngs using the argument set seed
            test_data = data_providers.EMNISTDataProvider('test', batch_size=1,
                                                          rng=rng,
                                                          flatten=False)  # initialize our rngs using the argument set seed
            num_output_classes = train_data.num_classes
            return {"train_data": train_data, "val_data": val_data, "test_data": test_data,"num_output_classes": num_output_classes}
        elif dataset_name == 'cifar10':
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
        
            trainset = data_providers.CIFAR10(root='data', set_name='train', download=True, transform=transform_train)
            train_data = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)
        
            valset = data_providers.CIFAR10(root='data', set_name='val', download=True, transform=transform_test)
            val_data = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=2)
        
            testset = data_providers.CIFAR10(root='data', set_name='test', download=True, transform=transform_test)
            test_data = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
        
            # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
            num_output_classes = 10
            return {"train_data": train_data, "val_data": val_data, "test_data": test_data,
                    "num_output_classes": num_output_classes}
        elif dataset_name == 'cifar100':
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
        
            trainset = data_providers.CIFAR100(root='data', set_name='train', download=True, transform=transform_train)
            train_data = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        
            valset = data_providers.CIFAR100(root='data', set_name='val', download=True, transform=transform_test)
            val_data = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=2)
        
            testset = data_providers.CIFAR100(root='data', set_name='test', download=True, transform=transform_test)
            test_data = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
        
            num_output_classes = 100
            return {"train_data": train_data, "val_data": val_data, "test_data": test_data, "num_output_classes": num_output_classes}
            
    def store_adv_examples(self):
        with open("adv_examples","w+") as f:
            json.dump(self.adv_examples, f)
        
    def get_adv_examples(self):
        return self.adv_examples

    def ResNetLoader(self, name, model_path):
        resnet = None
        if torch.cuda.is_available():
            fp = torch.load(model_path)
        else:
            fp = torch.load(model_path, map_location='cpu')
        assert (name == 'resnet50' or name == 'resnet101' or name == '')
        if name == 'resnet50':
            resnet = resnet50()
        if name == 'resnet101':
            resnet = resnet101()
        if name == "resnet152":
            resnet = resnet152()
        resnet.load_state_dict(fp['net'])
        resnet.eval()
        return resnet
    
    
class fgsm(attacks):
    stats = {'epsilon': [], 'accuracy': []}

    def __init__(self, epsilon, model_name,  dataset_name):
        super(fgsm, self).__init__(epsilon, model_name,  dataset_name)
        self.epsilon = epsilon
        self.experiment_name = model_name+"_"+dataset_name
        self.model_path = self.model_path + self.experiment_name+ "/" +self.experiment_name+".pwf"
        self.model = self.ResNetLoader(model_name,self.model_path)
        self.dataset = self.return_dataset(dataset_name)

    def attack(self):
        # Initialization
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model
        test_loader = self.dataset["test_data"]

        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model)
        model = model.to(device)
        for epsilon in self.epsilon:
            self.adv_examples = []
            print("attacking with epsilon:", epsilon)
            correct = 0
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data.requires_grad = True
            
                output = model(data)
                init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            
                # If the initial prediction is wrong, dont bother attacking, just move on
                if init_pred.item() != target.item():
                    continue
            
                loss = F.nll_loss(output, target)
                model.zero_grad()
                loss.backward()
            
                # Collect datagrad
                data_grad = data.grad.data
            
                perturbed_data = self.fgsm_attack(data, epsilon, data_grad)
            
                # Re-classify the perturbed image
                output = model(perturbed_data)
            
                # Check for success
                final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                if final_pred.item() == target.item():
                    correct += 1
                    # Special case for saving 0 epsilon examples
                    if (epsilon == 0) and (len( self.adv_examples) < 5):
                        adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                        self.adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
                else:
                    # Save some adv examples for visualization later
                    if len( self.adv_examples) < 5:
                        adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                        self.adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
            
                # Calculate final accuracy for this epsilon
            final_acc = correct / float(len(test_loader))
            self.stats['epsilon'].append(epsilon)
            self.stats['accuracy'].append(final_acc)
            print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))
            self.store_adv_examples()
        self.save_attack_statistic()
        return final_acc

    def save_attack_statistic(self):
        with open(self.stats_paths+self.experiment_name+'.csv', 'w+') as fp:
            json.dump(self.stats, fp)
    
        # Return the accuracy and an adversarial example

    def fgsm_attack(self,image, epsilon, data_grad):
        # print("Applying FGSM to image")
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
    
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + epsilon * sign_data_grad
    
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
        # Return the perturbed image
        return perturbed_image

att = fgsm(epsilon=[0.05,0.1,0.2,0.03],model_name="resnet50",dataset_name="cifar10")
att.attack()
att.store_adv_examples()


