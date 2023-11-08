import os
import copy
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
from oracle import *

import os
import pandas as pd
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pylab
import numpy as np
%load_ext autoreload
%autoreload 2


if __name__=='__main__':

    vae = get_vae(dataset_name='mnist', force_retrain=False)
    x_pool = MNIST(root='data/', train=True, transform=MNIST_transform(), 
                   download=True).data.view(-1,1,28,28).to(device).float()

    n_rep = 10
    qs_time = torch.zeros(n_rep)
    pool_time = torch.zeros(n_rep)
    qs_mean = torch.zeros(n_rep)
    qs_std = torch.zeros(n_rep)
    pool_mean = torch.zeros(n_rep)
    pool_std = torch.zeros(n_rep)
    batch_size = 4

    for i in range(n_rep):
        print("Starting rep: ",  i)
        model, _, _ = get_classifier(dataset_name='mnist',force_retrain=True)
        _, qs_batch = before_after_batch(model, vae, batch_size = batch_size)
        pool_batch = batchBALD(model,vae,4, x_pool)
        qs_mean[i], qs_std[i] = eval_batch(model,vae,qs_batch, num_samp_m = 100, num_samp_k = 100)
        pool_mean[i], pool_std[i] = eval_batch(model,vae,pool_batch, num_samp_m = 100, num_samp_k = 100)
        print("QS means: ",qs_mean)
        print("QS stds: ",qs_std)
        print("Pool means: ",pool_mean)
        print("Pool stds: ", pool_std)

    print(qs_mean.mean())
    print(pool_mean.mean())

    print((qs_mean-pool_mean).std()/np.sqrt(n_rep))




    df = pd.DataFrame("pool mean": [pool_mean.mean()], "qs mean": [qs_mean.mean()], "95% hw": [(qs_mean-pool_mean).std()/np.sqrt(n_rep)])
    df.to_csv("results/mnist_qs_vs_pool.csv")










    torch.set_default_dtype(torch.float32)
    model = Classifier('cifar').to(device)
    train_dataset = CIFAR10(root='data/', train=True, 
                                  transform=MNIST_transform(), download=True)

    x_data = (torch.from_numpy(train_dataset.data).to(float)/255.0-.5)*2
    x_data = x_data.to(device)
    x_data = x_data.type(torch.cuda.FloatTensor)
    x_data = x_data.permute(0, 3, 1,2)

    y_data = torch.Tensor(train_dataset.targets).to(int).to(device)
    del(train_dataset)


    num_gpu = 1 if torch.cuda.is_available() else 0

    # load the models
    from dcgan import Discriminator, Generator

    G = Generator(ngpu=1).eval()

    # load weights
    G.load_state_dict(torch.load('weights/netG_epoch_199.pth'))
    if torch.cuda.is_available():
        G = G.cuda()
    #G.decode = G


    def project(z,d=20):
        dists = torch.norm(z, p=2, dim=1).unsqueeze(dim=1)
        flags = dists>d
        return z*(flags.logical_not()) + z*flags*d/dists

    def maximize_func(func, input_tensor, other_params, lr = 0.01, num_steps = 20):
        # Create an optimizer object for Adam
        optimizer = optim.Adam([input_tensor], lr=lr)
        #optimizer = optim.SGD([input_tensor], lr=lr)
        # optimizer = optim.Adam([input_tensor.requires_grad_(True)], lr=0.01)

        # Optimize the input tensor to maximize the function
        for i in range(num_steps):
            optimizer.zero_grad()
            if other_params is None:
                output = func(input_tensor)
            else:
                output = func(input_tensor, *other_params)
            output.retain_grad()
            loss = -output
            loss.retain_grad()
            #print('next')
            print(-loss)
            loss.backward(retain_graph=True)
            #print(input_tensor.grad)
            optimizer.step()
            #optimizer.param_groups[0]['params'][0] = project(optimizer.param_groups[0]['params'][0])


        # Return the optimized input tensor
        return input_tensor


    @torch.enable_grad()
    def get_batch_utility_z_crn_cifar(z, g, model):
        g.eval()
        img = g(z)
        img.retain_grad()
        return get_batch_utility_crn(model, num_samp_m=10, 
                                 num_samp_k=10, x_data=img)

    def before_after_batch_cifar(model, g, batch_size = 16, lr = 1, num_steps = 100):
        latent_size = 100
        z = torch.randn(batch_size, latent_size, 1,1, device=device, requires_grad=True)
        z.retain_grad()
        before = g(z)
        found_z = maximize_func(func=get_batch_utility_z_crn_cifar, 
                                input_tensor = z,
                                num_steps=num_steps,
                                lr =lr,
                                other_params=(g,model))
        after = g(found_z)
        return before, after


    def batchBALD(model, vae, batch_size, x_data, num_samp_m = 5, num_samp_k = 5):
        assert batch_size<=len(x_data)
        scores = -np.inf*np.ones(len(x_data))
        indices = []
        for i in range(len(x_data)):
            scores[i] = get_utility(model, num_samp_k, x_data[i].unsqueeze(dim=0))
        indices.append(scores.argmax())
        while len(indices) < batch_size:
            scores = -np.inf*np.ones(len(x_data))
            for i in range(len(x_data)):
                if i not in indices: 
                    scores[i] = get_utility(model, num_samp_k, x_data[indices + [i]])
            indices.append(scores.argmax())
        return x_data[[indices]]

    def eval_batch(model,vae,img, num_samp_m = 10, num_samp_k = 10, num_sam = 10):
        vae.eval()
        sam_crn = [get_batch_utility_crn(model, num_samp_m=num_samp_m, num_samp_k=num_samp_k, x_data=img) for _ in range(num_sam)]
        sam_crn = torch.Tensor(sam_crn)
        mean_crn = torch.mean(sam_crn)
        std_crn = torch.std(sam_crn)
        return mean_crn, std_crn

    n_rep = 3
    qs_time = torch.zeros(n_rep)
    pool_time = torch.zeros(n_rep)
    qs_mean = torch.zeros(n_rep)
    qs_std = torch.zeros(n_rep)
    pool_mean = torch.zeros(n_rep)
    pool_std = torch.zeros(n_rep)
    batch_size = 2

    init_size = 100

    for i in range(n_rep):
        print("Starting rep: ",  i)
        indi = torch.randperm(x_data.size(0))[:init_size]
        x, y = x_data[indi].to(device), y_data[indi].to(device)
        model = Classifier('cifar').to(device)
        model = train_classifier_data(model, x,  y)
        _, qs_batch = before_after_batch_cifar(model, G, batch_size = batch_size, lr = .1, num_steps = 50)
        pool_batch = batchBALD(model, G, 2, x_data, num_samp_m = 2, num_samp_k = 2)
        qs_mean[i], qs_std[i] = eval_batch(model,G,qs_batch, num_samp_m = 150, num_samp_k = 150)
        pool_mean[i], pool_std[i] = eval_batch(model,G,pool_batch, num_samp_m = 150, num_samp_k = 150)
        print("QS means: ",qs_mean)
        print("QS stds: ",qs_std)
        print("Pool means: ",pool_mean)
        print("Pool stds: ", pool_std)

    print(qs_mean.mean())
    print(pool_mean.mean())

    print((qs_mean-pool_mean).std()/np.sqrt(n_rep))

    df = pd.DataFrame("pool mean": [pool_mean.mean()], "qs mean": [qs_mean.mean()], "95% hw": [(qs_mean-pool_mean).std()/np.sqrt(n_rep)])
    df.to_csv("results/cifar_qs_vs_pool.csv")