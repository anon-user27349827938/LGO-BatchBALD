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


    torch.set_default_dtype(torch.float32)
    model = Classifier('cifar').to(device)
    train_dataset = CIFAR10(root='data/', train=True, 
                                  transform=MNIST_transform(), download=True)
    x_data = (torch.from_numpy(train_dataset.data).to(float)/255.0-.5)*2
    x_data = x_data.to(device)
    x_data = x_data.type(torch.cuda.FloatTensor)
    x_data = x_data.permute(0, 3, 1,2)
    y_data = torch.Tensor(train_dataset.targets).to(int).to(device)
    init_size = 2000
    x_data, y_data = x_data[0:init_size],  y_data[0:init_size]
    del(train_dataset)
    model = train_classifier_data(model, x_data,  y_data)


    def eval_model(model, x_data, y_data):
        correct = 0
        total = 0
        for _ in range(10): # getting multiple samples from ENN
            flag = True
            if flag:
                outputs = model(x_data).to(device)
                flag = False
            else:
                outputs += model(x_data).to(device)
            preds = outputs.argmax(dim=1)
            correct += (preds==y_data).sum()
            total += len(y_data)
        return correct/total

    acc = eval_model(model,x_data, y_data)
    acc

    num_gpu = 1 if torch.cuda.is_available() else 0

    # load the models
    from dcgan import Discriminator, Generator

    D = Discriminator(ngpu=1).eval()
    G = Generator(ngpu=1).eval()

    # load weights
    D.load_state_dict(torch.load('weights/netD_epoch_199.pth'))
    G.load_state_dict(torch.load('weights/netG_epoch_199.pth'))
    if torch.cuda.is_available():
        D = D.cuda()
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

    def before_after_batch_cifar(model, batch_size = 16, lr = 1, num_steps = 100):
        batch = torch.randn(batch_size, 3, 32,32, device=device, requires_grad=True)
        after = maximize_func(func=get_batch_utility_crn, 
                                input_tensor = batch,
                                num_steps=num_steps,
                                lr =lr,
                                other_params=(model))
        return before, after

    @torch.enable_grad()
    def new_signature_new_est(x_data, model, num_samp_m, num_samp_k):
        return get_batch_utility_crn(model, num_samp_m, num_samp_k, x_data)

    batch = torch.randn(8, 3, 32, 32, device=device, requires_grad=True)
    before = batch + 0
    found_batch = maximize_func(func=new_signature_new_est, input_tensor = batch, num_steps=100, lr =1, other_params=(model,10,10))
    def plot_batches_in_rows_cifar(batch1, batch2):
        num_images = len(batch1)
        batch1 = batch1.cpu().detach().numpy()
        batch1 = batch1.reshape(num_images, 3, 32, 32)
        batch1 = batch1.transpose((0, 2, 3, 1))
        batch2 = batch2.cpu().detach().numpy()
        batch2 = batch2.reshape(num_images, 3, 32, 32)
        batch2 = batch2.transpose((0, 2, 3, 1))
        # Create a figure and axis for plotting
        fig, ax = plt.subplots(2, num_images, figsize=(num_images, 2))

        # Iterate through the images in the first batch and plot them
        for i in range(num_images):
            ax[0, i].imshow((batch1[i]+1)/2.0, interpolation='bilinear')
            ax[0, i].axis('off')

        # Iterate through the images in the second batch and plot them
        for i in range(num_images):
            ax[1, i].imshow((batch2[i]+1)/2.0, interpolation='bilinear')
            ax[1, i].axis('off')

        # Display the rows of images
        plt.show()

    plot_batches_in_rows_cifar(before,found_batch, "results/cifar_x_visual.png")



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

    before, after = before_after_batch_cifar(model, G, batch_size = 8, lr = .1, num_steps = 50)

    def plot_batches_in_rows_cifar(batch1, batch2, path):
        num_images = len(batch1)
        batch1 = batch1.cpu().detach().numpy()
        batch1 = batch1.reshape(num_images, 3, 32, 32)
        batch1 = batch1.transpose((0, 2, 3, 1))
        batch2 = batch2.cpu().detach().numpy()
        batch2 = batch2.reshape(num_images, 3, 32, 32)
        batch2 = batch2.transpose((0, 2, 3, 1))
        # Create a figure and axis for plotting
        fig, ax = plt.subplots(2, num_images, figsize=(num_images, 2))

        # Iterate through the images in the first batch and plot them
        for i in range(num_images):
            ax[0, i].imshow((batch1[i]+1)/2.0, interpolation='bilinear')
            ax[0, i].axis('off')

        # Iterate through the images in the second batch and plot them
        for i in range(num_images):
            ax[1, i].imshow((batch2[i]+1)/2.0, interpolation='bilinear')
            ax[1, i].axis('off')

        # Display the rows of images
        plt.show()
        plt.savefig(path)


    plot_batches_in_rows_cifar(before,after,"results/cifar_z_visual.png")




















    _, test_loader = get_train_test_loader(dataset_name='mnist', 
                                                        train_batch_size=classifier_train_batch_size,
                                                        test_batch_size=classifier_test_batch_size)

    vae = get_vae(dataset_name='mnist', force_retrain=False)
    model, _, _ = get_classifier(dataset_name='mnist',force_retrain=True)
    before, after = before_after_batch(model, vae, batch_size = 8, lr = 1, num_steps = 100)

    plot_batches_in_rows(before, after,"results/mnist_z_visual.png")


    @torch.enable_grad()
    def new_signature_new_est(x_data, model, num_samp_m, num_samp_k):
        return get_batch_utility_crn(model, num_samp_m, num_samp_k, x_data)

    batch = torch.randn(8, 1, 28, 28, device=device, requires_grad=True)
    batch_copy = batch + 0
    found_batch = maximize_func(func=new_signature_new_est, input_tensor = batch, num_steps=100, lr =1, other_params=(model,10,10))

    plot_batches_in_rows(batch_copy, found_batch, "results/mnist_x_visual.png")