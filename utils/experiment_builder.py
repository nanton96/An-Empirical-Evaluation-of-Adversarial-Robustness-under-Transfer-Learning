import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import os
import numpy as np
import time
import random
from scipy.stats import truncnorm
from utils.storage_utils import save_statistics
from utils.utils import pred_batch,to_var
from utils.train import adv_train, FGSM_train_rnd
from utils.attacks import FGSMAttack, LinfPGDAttack
# import GPUtil
import copy

class ExperimentBuilder(nn.Module):

    def __init__(self, network_model, experiment_name, num_epochs, train_data, val_data,
                 test_data, weight_decay_coefficient, use_gpu, scheduler, optimizer,device, adversary ='fgsm', continue_from_epoch=-1, adv_train= False ):
        """
        Initializes an ExperimentBuilder object. Such an object takes care of running training and evaluation of a deep net
        on a given dataset. It also takes care of saving per epoch models and automatically inferring the best val model
        to be used for evaluating the test set metrics.
        :param network_model: A pytorch nn.Module which implements a network architecture.
        :param experiment_name: The name of the experiment. This is used mainly for keeping track of the experiment and creating and directory structure that will be used to save logs, model parameters and other.
        :param num_epochs: Total number of epochs to run the experiment
        :param train_data: An object of the DataProvider type. Contains the training set.
        :param val_data: An object of the DataProvider type. Contains the val set.
        :param test_data: An object of the DataProvider type. Contains the test set.
        :param weight_decay_coefficient: A float indicating the weight decay to use with the adam optimizer.
        :param use_gpu: A boolean indicating whether to use a GPU or not.
        :param continue_from_epoch: An int indicating whether we'll start from scrach (-1) or whether we'll reload a previously saved model of epoch 'continue_from_epoch' and continue training from there.
        """
        super(ExperimentBuilder, self).__init__()
        # if torch.cuda.is_available() and use_gpu:  # checks whether a cuda gpu is available and whether the gpu flag is True
        #     if "," in gpu_id:
        #         self.device = [torch.device('cuda:{}'.format(idx)) for idx in gpu_id.split(",")]  # sets device to be cuda
        #     else:
        #         self.device = torch.device('cuda:{}'.format(gpu_id))  # sets device to be cuda

        #     os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id  # sets the main GPU to be the one at index 0 (on multi gpu machines you can choose which one you want to use by using the relevant GPU ID)
        #     print("use GPU")
        #     print("GPU ID {}".format(gpu_id))
        # else:
        #     print("use CPU")
        #     self.device = torch.device('cpu')  # sets the device to be CPU

        self.adv_train = adv_train
        if adv_train:
            if adversary == 'fgsm':
                self.attacker = FGSMAttack
            elif adversary == 'pgd':
                self.attacker = LinfPGDAttack
        else:
            self.attacker = None
        self.device = device
        self.delay = 15
        self.experiment_name = experiment_name
        self.model = network_model
        #self.model.reset_parameters()
        
        # Create truncated normal distribution - For details check https://arxiv.org/pdf/1611.01236.pdf
        lower, upper,mu, sigma = 0, 0.125,0, 0.0625
        self.distribution = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        
        
        if torch.cuda.device_count() > 1:
            self.model.to(self.device)
            self.model = nn.DataParallel(module=self.model)
            
        else:
            self.model.to(self.device)  # sends the model from the cpu to the gpu
          # re-initialize network parameters
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        # Generate the directory names
        self.experiment_folder = os.path.abspath("experiments_results/"+experiment_name)
        self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))
        self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))
        print(self.experiment_folder, self.experiment_logs)
        # Set best models to be at 0 since we are just starting
        self.best_val_model_idx = 0
        self.best_val_model_acc = 0.

        if not os.path.exists("experiments_results"):  # If experiment directory does not exist
            os.mkdir("experiments_results")

        if not os.path.exists(self.experiment_folder):  # If experiment directory does not exist
            os.mkdir(self.experiment_folder)  # create the experiment directory

        if not os.path.exists(self.experiment_logs):
            os.mkdir(self.experiment_logs)  # create the experiment log directory

        if not os.path.exists(self.experiment_saved_models):
            os.mkdir(self.experiment_saved_models)  # create the experiment saved models directory

        self.num_epochs = num_epochs
        self.criterion = nn.CrossEntropyLoss().to(self.device)  # send the loss computation to the GPU
        if continue_from_epoch == -2:
            try:
                self.best_val_model_idx, self.best_val_model_acc, self.state = self.load_model(
                    model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                    model_idx='latest')  # reload existing model from epoch and return best val model index
                # and the best val acc of that model
                self.starting_epoch = continue_from_epoch
            except:
                print("Model objects cannot be found, initializing a new model and starting from scratch")
                self.starting_epoch = 0
                self.state = dict()

        elif continue_from_epoch != -1:  # if continue from epoch is not -1 then
            self.best_val_model_idx, self.best_val_model_acc, self.state = self.load_model(
                model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                model_idx=continue_from_epoch)  # reload existing model from epoch and return best val model index
            # and the best val acc of that model
            self.starting_epoch = self.state['current_epoch_idx']
        else:
            self.starting_epoch = 0
            self.state = dict()

    def get_num_parameters(self):
        total_num_params = 0
        for param in self.parameters():
            total_num_params += np.prod(param.shape)

        return total_num_params

    def run_train_iter(self, x, y):
        """
        Receives the inputs and targets for the model and runs a training iteration. Returns loss and accuracy metrics.
        :param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
        :param y: The targets for the model. A numpy array of shape batch_size, num_classes
        :return: the loss and accuracy for this batch
        """
        self.train()  # sets model to training mode (in case batch normalization or other methods have different procedures for training and evaluation)

        if len(y.shape) > 1:
            y = np.argmax(y, axis=1)  # convert one hot encoded labels to single integer labels

        #print(type(x))

        if type(x) is np.ndarray:
            x, y = torch.Tensor(x).float().to(device=self.device), torch.Tensor(y).long().to(
            device=self.device)  # send data to device as torch tensors

        x = x.to(self.device)
        y = y.to(self.device)
        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0

        out = self.model.forward(x)  # forward the data in the model
        loss = F.cross_entropy(input=out, target=y)  # compute loss

        loss.backward()             # backpropagate to compute gradients for current iter loss

        self.optimizer.step()  # update network parameters
        _, predicted = torch.max(out.data, 1)  # get argmax of predictions
        accuracy = np.mean(list(predicted.eq(y.data).cpu()))  # compute accuracy
        return loss.data.detach().cpu().numpy(), accuracy

    def run_evaluation_iter(self, x, y):

        """
        Receives the inputs and targets for the model and runs an evaluation iterations. Returns loss and accuracy metrics.
        :param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
        :param y: The targets for the model. A numpy array of shape batch_size, num_classes
        :return: the loss and accuracy for this batch
        """
        self.eval()  # sets the system to validation mode
        if len(y.shape) > 1:
            y = np.argmax(y, axis=1)  # convert one hot encoded labels to single integer labels
        if type(x) is np.ndarray:
            x, y = torch.Tensor(x).float().to(device=self.device), torch.Tensor(y).long().to(
            device=self.device)  # convert data to pytorch tensors and send to the computation device

        x = x.to(self.device)
        y = y.to(self.device)
        out = self.model.forward(x)  # forward the data in the model
        loss = F.cross_entropy(out, y)  # compute loss
        _, predicted = torch.max(out.data, 1)  # get argmax of predictions
        accuracy = np.mean(list(predicted.eq(y.data).cpu()))  # compute accuracy
        return loss.data.detach().cpu().numpy(), accuracy

    def run_adv_train_iter(self,x,y):
              
        self.train()
        train_stat = {
            'clean_acc':0,
            'clean_loss': 0,
            'adv_acc':0,
            'adv_loss': 0
        }
        # convert one hot encoded labels to single integer labels
        if len(y.shape) > 1:
            y = np.argmax(y, axis=1)               

        if type(x) is np.ndarray:
            x, y = torch.Tensor(x).float().to(device=self.device), torch.Tensor(y).long().to(
            device=self.device)                     # send data to device as torch tensors

        x = x.to(self.device)
        y = y.to(self.device)     

        # First half of the attack - Clean examples accuracy 
        out = self.model(x)
        _,predicted = torch.max(out.data, 1)  
        accuracy = np.mean(list(predicted.eq(y.data).cpu()))
        loss = F.cross_entropy(input=out, target=y)  # compute loss
        train_stat['clean_acc'] = accuracy
        train_stat['clean_loss'] = np.asscalar(loss.data.detach().cpu().numpy())


        # Prevent label leaking, by using most probable state
        y_pred  = pred_batch(x,self.model)


        # Create corresponding adversarial examples for training 

        e = self.distribution.rvs(1)[0]
        advesary =  self.attacker(epsilon = e)
        x_adv = adv_train(x,y_pred, self.model,F.cross_entropy,advesary)
        x_adv_var = to_var(x_adv)
        out = self.model(x_adv_var)
        _,predicted = torch.max(out.data, 1)  
        adv_acc = np.mean(list(predicted.eq(y.data).cpu()))
        
        loss_adv =  F.cross_entropy(out, y.data)
        train_stat['adv_acc'] = adv_acc
        train_stat['adv_loss'] = np.asscalar(loss_adv.data.detach().cpu().numpy())

        loss = (loss + loss_adv) / 2
        accuracy =  (accuracy + adv_acc)/2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()  

        # GPUtil.showUtilization()
        # GPUtil.showUtilization()
        return np.asscalar(loss.data.detach().cpu().numpy()), accuracy, train_stat

    def run_adv_evaluation_iter(self,x,y):
      
        # ---------------- Testing the given network --------------- #

        self.eval()  # sets the system to validation mode

        validaton_stat = {
        'clean_acc':0,
        'clean_loss': 0,
        'adv_acc':0,
        'adv_loss': 0
        }

        if len(y.shape) > 1:
            y = np.argmax(y, axis=1)  # convert one hot encoded labels to single integer labels
        if type(x) is np.ndarray:
            x, y = torch.Tensor(x).float().to(device=self.device), torch.Tensor(y).long().to(
            device=self.device)  # convert data to pytorch tensors and send to the computation device
        x = x.to(self.device)
        y = y.to(self.device)


        out = self.model(x)
        loss = F.cross_entropy(input=out, target=y)
        _,predicted = torch.max(out.data, 1)  
        accuracy = np.mean(list(predicted.eq(y.data).cpu()))
        validaton_stat['clean_acc']  = accuracy
        validaton_stat['clean_loss'] = np.asscalar(loss.data.detach().cpu().numpy())
        
        # Prevent label leaking, by using most probable state

        y_pred  = pred_batch(x,self.model)

        # Create corresponding adversarial examples for training 

        e = self.distribution.rvs(1)[0]
        adversary = self.attacker(epsilon = e)
        x_adv = adv_train(x,y_pred, self.model,F.cross_entropy,adversary) 
        x_adv_var = to_var(x_adv)
        out = self.model(x_adv_var)
        _,predicted = torch.max(out.data, 1)  
        adv_acc = np.mean(list(predicted.eq(y.data).cpu()))
        loss_adv =  F.cross_entropy(out, y.data)

        validaton_stat['adv_acc']  = adv_acc
        validaton_stat['adv_loss'] = np.asscalar(loss_adv.data.detach().cpu().numpy())

        loss = (loss + loss_adv) / 2   
        accuracy =  (accuracy + adv_acc)/2

        # GPUtil.showUtilization()
        return np.asscalar(loss.data.detach().cpu().numpy()), accuracy, validaton_stat

    def save_model(self, model_save_dir, model_save_name, model_idx, state):
        """
        Save the network parameter state and current best val epoch idx and best val accuracy.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :param best_validation_model_idx: The index of the best validation model to be stored for future use.
        :param best_validation_model_acc: The best validation accuracy to be stored for use at test time.
        :param model_save_dir: The directory to store the state at.
        :param state: The dictionary containing the system state.

        """
        state['network'] = self.best_val_model  # save network parameter and other variables.
        torch.save(state, f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(
            model_idx))))  # save state at prespecified filepath

    def save_readable_model(self, model_save_dir, state_dict):
        state ={'network': state_dict} # save network parameter and other variables.
        fname = os.path.join(model_save_dir, "train_model_best_readable")
        print('Saving state in ', fname)
        torch.save(state, f=fname)  # save state at prespecified filepath

    def load_model(self, model_save_dir, model_save_name, model_idx):
        """
        Load the network parameter state and the best val model idx and best val acc to be compared with the future val accuracies, in order to choose the best val model
        :param model_save_dir: The directory to store the state at.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :return: best val idx and best val model acc, also it loads the network state into the system state without returning it
        """
        state = torch.load(f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx))))
        self.load_state_dict(state_dict=state['network'])
        return state['best_val_model_idx'], state['best_val_model_acc'], state

    def run_experiment(self):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model and val model accuracy after each epoch
        :return: The summary current_epoch_losses from starting epoch to total_epochs.
        """
        print("adversarial training flag is:",self.adv_train)
        if self.adv_train:
            total_losses = {"clean_train_acc":[], "adv_train_acc":[], "clean_train_loss":[], "adv_train_loss":[], "train_acc": [], "train_loss": [],
                            "clean_val_acc":[], "adv_val_acc":[], "clean_val_loss":[], "adv_val_loss":[], "val_acc": [], "val_loss": [],
                             "curr_epoch": []}  # initialize a dict to keep the per-epoch metrics
        else:           
            total_losses = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": [], "curr_epoch": []}  # initialize a dict to keep the per-epoch metrics

        for i, epoch_idx in enumerate(range(self.starting_epoch, self.num_epochs)):
            epoch_start_time = time.time()
            if self.adv_train:
                current_epoch_losses = {"clean_train_acc":[], "adv_train_acc":[], "clean_train_loss":[], "adv_train_loss":[], "train_acc": [], "train_loss": [],
                            "clean_val_acc":[], "adv_val_acc":[], "clean_val_loss":[], "adv_val_loss":[], "val_acc": [], "val_loss": []}
            else:
                current_epoch_losses = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []}

            with tqdm.tqdm(total=len(self.train_data)) as pbar_train:  # create a progress bar for training
                for idx, (x, y) in enumerate(self.train_data):         # get data batches
                    if self.adv_train:
                        loss,accuracy,train_stat = self.run_adv_train_iter(x=x, y=y)  # take a training iter step
                        current_epoch_losses["clean_train_acc"].append(train_stat['clean_acc']) 
                        current_epoch_losses["adv_train_acc"].append(train_stat['adv_acc']) 
                        current_epoch_losses["clean_train_loss"].append(train_stat['clean_loss']) 
                        current_epoch_losses["adv_train_loss"].append(train_stat[ 'adv_loss'])                         
                    else:
                       loss, accuracy = self.run_train_iter(x=x, y=y)  # take a training iter step
                

                    current_epoch_losses["train_loss"].append(loss)         # add current iter loss to the train loss list
                    current_epoch_losses["train_acc"].append(accuracy)      # add current iter acc to the train acc list
                                           
                    pbar_train.update(1)
                    pbar_train.set_description("train loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))

            with tqdm.tqdm(total=len(self.val_data)) as pbar_val:  # create a progress bar for validation
                for x, y in self.val_data:  # get data batches
                    if self.adv_train :
                        loss, accuracy,val_stat = self.run_adv_evaluation_iter(x=x, y=y)  # run a validation iter
                        current_epoch_losses["clean_val_acc"].append(val_stat['clean_acc']) 
                        current_epoch_losses["adv_val_acc"].append(val_stat['adv_acc']) 
                        current_epoch_losses["clean_val_loss"].append(val_stat['clean_loss']) 
                        current_epoch_losses["adv_val_loss"].append(val_stat[ 'adv_loss']) 
                    else:
                        loss, accuracy = self.run_evaluation_iter(x=x, y=y)  # run a validation iter                      

                    current_epoch_losses["val_loss"].append(loss)  # add current iter loss to val loss list.
                    current_epoch_losses["val_acc"].append(accuracy)  # add current iter acc to val acc lst.
                    pbar_val.update(1)  # add 1 step to the progress bar
                    pbar_val.set_description("val loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))
            val_mean_accuracy = np.mean(current_epoch_losses['val_acc'])
            if val_mean_accuracy > self.best_val_model_acc:  # if current epoch's mean val acc is greater than the saved best val acc then
                self.best_val_model_acc = val_mean_accuracy  # set the best val model acc to be current epoch's val accuracy
                self.best_val_model_idx = epoch_idx  # set the experiment-wise best val idx to be the current epoch's idx
                self.best_val_model = copy.deepcopy(self.state_dict()) 
            for key, value in current_epoch_losses.items():
                total_losses[key].append(np.mean(value))  # get mean of all metrics of current epoch metrics dict, to get them ready for storage and output on the terminal.

            total_losses['curr_epoch'].append(epoch_idx)
            save_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv', stats_dict=total_losses, current_epoch=epoch_idx,
                            continue_from_mode=True if (self.starting_epoch != 0 or i > 0) else False) # save statistics to stats file.

            # load_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv') # How to load a csv file if you need to

            out_string = "_".join(
                ["{}_{:.4f}".format(key, np.mean(value)) for key, value in current_epoch_losses.items()])
            # create a string to use to report our epoch metrics
            epoch_elapsed_time = time.time() - epoch_start_time  # calculate time taken for epoch
            epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
            print("Epoch {}:".format(epoch_idx), out_string, "epoch time", epoch_elapsed_time, "seconds")
            self.state['current_epoch_idx'] = epoch_idx
            self.state['best_val_model_acc'] = self.best_val_model_acc
            self.state['best_val_model_idx'] = self.best_val_model_idx
            #update scheduler
            self.scheduler.step()

            self.save_model(model_save_dir=self.experiment_saved_models,
                            model_save_name="train_model", model_idx='latest', state=self.state)


        print("Generating test set evaluation metrics")
        self.save_model(model_save_dir=self.experiment_saved_models, model_save_name="train_model", model_idx="best", state=self.state)
        self.load_model(model_save_dir=self.experiment_saved_models, model_save_name="train_model", model_idx="best")

        # Save a generic readable model format
        try:
            state_dict = self.model.module.state_dict()
        except AttributeError:
            state_dict = self.model.state_dict()
        self.save_readable_model(self.experiment_saved_models, state_dict)

        current_epoch_losses = {"test_acc": [], "test_loss": []}  # initialize a statistics dict
        with tqdm.tqdm(total=len(self.test_data)) as pbar_test:  # ini a progress bar
            for x, y in self.test_data:  # sample batch
                loss, accuracy = self.run_evaluation_iter(x=x,
                                                          y=y)  # compute loss and accuracy by running an evaluation step
                current_epoch_losses["test_loss"].append(loss)  # save test loss
                current_epoch_losses["test_acc"].append(accuracy)  # save test accuracy
                pbar_test.update(1)  # update progress bar status
                pbar_test.set_description(
                    "loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))  # update progress bar string output

        test_losses = {key: [np.mean(value)] for key, value in
                       current_epoch_losses.items()}  # save test set metrics in dict format
        save_statistics(experiment_log_dir=self.experiment_logs, filename='test_summary.csv',
                        # save test set metrics on disk in .csv format
                        stats_dict=test_losses, current_epoch=0, continue_from_mode=False)

        return total_losses, test_losses

  