import os
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np
# https://github.com/arturml/mnist-cgan

# %matplotlib inline
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder, MNIST
from torchvision import transforms
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import make_grid
from torch.autograd import grad
from tqdm import tqdm

# Self defined modules
from EpiNet import *
from batchBALD import *
from params import *
from get_data import *
from generative import *
from misc_plotting import *

# This is the same as the generative.py conv 
def _conv(channel_size, kernel_num):
    return nn.Sequential(
        nn.Conv2d(
            channel_size, kernel_num,
            kernel_size=4, stride=2, padding=1,
        ),
        nn.BatchNorm2d(kernel_num),
        nn.ReLU(),
    )

# Making this Flatten module here so that the x.view(-1) can be within the nn.Sequential
# This is copied from https://discuss.pytorch.org/t/how-to-build-a-view-layer-in-pytorch-for-sequential-models/53958/5
class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        '''
        Note that input.size(0) is usually the batch size.
        So what it does is that given any input with input.size(0) # of batches,
        will flatten to be 1 * nb_elements.
        '''
        batch_size = input.size(0)
        out = input.view(batch_size,-1)
        return out # (batch_size, *size)
    
class Oracle(nn.Module):
    def __init__(self, dataset_name='mnist'):
        super().__init__()
        self.dataset_name = dataset_name

        if self.dataset_name == 'mnist':
            self.base = nn.Sequential(
                nn.Linear(784, 1024),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(1024, 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True)
            ).to(device)
        elif self.dataset_name == 'cifar':
            self.channel_num = 3
            self.kernel_num = 128
            self.image_size = 32

            self.base = nn.Sequential(
                _conv(self.channel_num, self.kernel_num // 4),
                _conv(self.kernel_num // 4, self.kernel_num // 2),
                _conv(self.kernel_num // 2, self.kernel_num),
                Flatten(),
                nn.Linear(2048, 256)
            ).to(device)
        
        self.softmax = nn.Softmax().to(device)
        self.linear = nn.Linear(256, 10).to(device)
    
    def forward(self, x, z = None):
        if self.dataset_name == 'mnist':
            x = x.view(x.size(0), 784).to(device)
        features = self.base(x).to(device) # features extracted by next to last layer of base nn
        scores = self.linear(features).to(device)
        out = self.softmax(scores-torch.max(scores,dim=1)[0].unsqueeze(dim=1).to(device))
        out = out.squeeze()
        return out.to(device)


def get_oracle(dataset_name='mnist', force_retrain=False):  
    '''
    creates and trains classifier
    '''
    # If we can load the classifier model, just load that
    save_ckpt_file = '{}_oracle.pth'.format(dataset_name)
    if os.path.exists(save_ckpt_file) and not force_retrain:
        model = Oracle(dataset_name).to(device)
        model.load_state_dict(torch.load(save_ckpt_file))
        return model.to(device)

    full_loader = get_full_loader(dataset_name=dataset_name, 
                                                    batch_size=oracle_train_batch_size)

    model = Oracle(dataset_name).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    #pyro.clear_param_store()
    for epoch in range(num_epochs):
        for i, (x_data, y_data) in enumerate(full_loader):
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            # calculate the loss and take a gradient step
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(x_data).to(device)
            loss = criterion(outputs, y_data)
            loss.backward()
            optimizer.step()
            # Print progress
            if (i) % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], \
                      Step [{i+1}/{len(full_loader)}], \
                      Loss: {loss.item():.4f}')
    
    torch.save(model.state_dict(), '{}_oracle.pth'.format(dataset_name))
    return model

