# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester
# Modified by Joao Marques (jmc12) for the fall 2021 semester 

"""
This is the main entry point for MP3. You should only modify code
within this file, neuralnet_learderboard and neuralnet_part2.py -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.pooling import MaxPool2d
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader

class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension
        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.lrate = lrate
        self.in_size = in_size
        self.out_size = out_size
        self.model = nn.Sequential(
            # nn.Linear(in_size,200),
            # nn.LeakyReLU(),
            # nn.Dropout(p=0.03),
            # nn.Linear(200,out_size)
            nn.Conv2d(in_channels=3,out_channels=12,kernel_size=4,stride=1,padding=1),
            nn.MaxPool2d(kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout(p=0.02),
            nn.Conv2d(in_channels=12,out_channels=36,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(),
            nn.Dropout(p=0.02),
            nn.MaxPool2d(kernel_size=3),
            # nn.Conv2d(in_channels=48,out_channels=96,kernel_size=3,stride=1,padding=1),
            # nn.MaxPool2d(kernel_size=3),
            # nn.LeakyReLU(),
            # nn.Dropout(p=0.01),
            nn.Flatten(),
            nn.Linear(324,out_size)

        )
        
        self.opt = torch.optim.SGD(self.parameters(), self.lrate, momentum=0.9)
    

    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        # torch.ones(x.shape[0], 1)
        reshape = x.view(x.shape[0],3,32,32)
        reshape_norm = reshape
        for i in range(x.shape[0]):
            reshape_norm[i] = (reshape[i]-torch.mean(reshape[i])) / torch.std(reshape[i])
        out = self.model(reshape_norm)
        return out

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        opt = self.opt
        opt.zero_grad()
        loss_fn = self.loss_fn
        yhat = self.forward(x)
        backward_loss = loss_fn(yhat,y)
        backward_loss.backward()
        opt.step()
        # breakpoint()
        return backward_loss.item()



def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method _must_ work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of floats containing the total loss at the beginning and after each epoch.
            Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    # return a list of the losses for each epoch of training, a numpy array with the estimated 
    # class labels (0, 1, 2, or 3) for the dev set and the trained network
    criterion = nn.CrossEntropyLoss()
    in_size = 32*32*3
    out_size = 4
    n_n = NeuralNet(0.02,criterion,in_size,out_size)
    
    num_pic = len(train_labels)
    num_test = len(dev_set)
    
    np_train_set = np.array(train_set)

    # standarize
    # std_list = (train_set-train_set.mean())/train_set.std()
    for idx in range(num_pic):
        mean = np.mean(np_train_set[idx])
        std = np.std(np_train_set[idx])
        np_train_set[idx] = (np_train_set[idx]-mean)/std
    # breakpoint()
    std_list = np_train_set

    # extend list to prevent out of bound
    loss_list = []
    for j in range(epochs):
        for i in range(num_pic//batch_size):
            st_idx = i*batch_size%num_pic
            ed_idx = st_idx+batch_size

            subset = std_list[st_idx:ed_idx]
            sublabel = train_labels[st_idx:ed_idx]
            # if (i == 22):
            #     breakpoint
            
            # normalization
            m = nn.BatchNorm1d(3072)

            back_loss = n_n.step(torch.tensor(subset),sublabel)
            loss_list.append(back_loss)

    # print(loss_list)
    pred_list = n_n.forward(dev_set)
    pred_no_tensor = pred_list.tolist()
    out = []
    for j in range(num_test):
        max_index = pred_no_tensor[j].index(max(pred_no_tensor[j]))
        out.append(max_index)
    loss_list = np.array(loss_list)
    # breakpoint()
    out = np.array(out)
    return list(loss_list),out,n_n