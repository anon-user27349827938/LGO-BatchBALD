
import torch
import torch.nn as nn
import math, os
import numpy as np

from params import *
from get_data import *

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
    
class InvertibleLinearLayer(nn.Module):
    # class written by chatgpt
    def __init__(self, in_features=10, out_features=10):
        super(InvertibleLinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        output = nn.functional.linear(x, self.weight, self.bias)
        return output

    def inverse(self, y):
        # Compute the inverse of the linear transformation
        inv_weight = torch.inverse(self.weight)
        inv_bias = -torch.matmul(inv_weight, self.bias)
        x = nn.functional.linear(y, inv_weight, inv_bias)
        return x

    def log_det_jacobian(self, x):
        # Compute the log-determinant of the Jacobian
        return torch.slogdet(self.weight)[1] * x.size(-1)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

class Classifier(nn.Module):
    def __init__(self, dataset_name='mnist'):
        super().__init__()
        self.zdim = 1000
        self.dataset_name = dataset_name

        if self.dataset_name == 'mnist':
            self.base = nn.Sequential(
                nn.Linear(784, 1024),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(1024, 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True)
            ) 
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
            )
        
        self.epi_learnable = nn.Sequential(
            nn.Linear(256 + self.zdim, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 10))
        
        self.epi_prior = nn.Sequential(
            nn.Linear(256 + self.zdim, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 10))
        self.softmax = nn.Softmax()
        self.linear = nn.Linear(256, 10)
    
    
    @torch.enable_grad()
    def forward(self, x, z = None):
        if self.dataset_name == 'mnist':
            x = x.view(x.size(0), 784)
        features = self.base(x) # features extracted by next to last layer of base nn
        scores = self.linear(features)
        if z == None:
            z = torch.randn(1, self.zdim,  device=device)
            
        # implement stop gradient on features with detach, sample z and concatenate it to features for each x        
        epi_input = torch.cat([features.detach(), z.repeat(x.shape[0],1)],dim=1)
        scores += self.epi_learnable(epi_input) + self.epi_prior(epi_input).detach()
               
        out = self.softmax(scores-torch.max(scores,dim=1)[0].unsqueeze(dim=1).to(device))
        out = out.squeeze()
        
        '''
        if return_p: # this part will likely be useful for soft feedback as prob dists
            logp = self.epinet_z.log_det_jacobian(z) + torch.distributions.Normal(loc = 0, scale = 1).log_prob(z).sum()
            return out, torch.exp(logp)
        '''
        
        return out


def train_classifier_data(model, x_data, y_data):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_iterations = 500
    #pyro.clear_param_store()
    for j in range(num_iterations):
        # calculate the loss and take a gradient step
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(x_data)
        loss = criterion(outputs, y_data)
        loss.backward(retain_graph=True)
        optimizer.step()
        if j % 10 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(y_data)))
    return model

def get_classifier(dataset_name='mnist', force_retrain=False):  
    '''
    creates and trains classifier
    '''
    # If we can load the classifier model, just load that
    save_ckpt_file = '{}_epinet.pth'.format(dataset_name)
    if os.path.exists(save_ckpt_file) and not force_retrain:
        model = Classifier(dataset_name).to(device)
        model.load_state_dict(torch.load(save_ckpt_file))
        return model.to(device)

    data_loader, test_loader = get_train_test_loader(dataset_name=dataset_name, 
                                                    train_batch_size=classifier_train_batch_size,
                                                    test_batch_size=classifier_train_batch_size)

    (x_data, y_data) = next(iter(data_loader))
    x_data = x_data.to(device)
    y_data = y_data.to(device)

    model = Classifier(dataset_name).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_iterations = 500
    #pyro.clear_param_store()
    for j in range(num_iterations):
        # calculate the loss and take a gradient step
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(x_data)
        loss = criterion(outputs, y_data)
        loss.backward(retain_graph=True)
        optimizer.step()
        if j % 10 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(y_data)))
    
    torch.save(model.state_dict(), '{}_epinet.pth'.format(dataset_name))
    return model, x_data, y_data

if __name__ == '__main__':
    get_classifier('cifar', force_retrain=True)

