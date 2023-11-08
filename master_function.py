import os
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import math
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


if __name__ == '__main__':
    model = get_classifier(dataset_name='mnist').to(device)
    vae = load_vae(dataset_name='mnist').to(device)
    print(model)
    print(vae)

    ##################
    # z optimization #
    ##################
    # z = torch.randn(find_z_batch_size, latent_size, requires_grad=True)
    # # z = Variable(z, requires_grad=True)
    # max_dist_train = np.inf # currently makes constraint inactive, (torch.norm(x_data,p=2,dim=1)).max()
    # found_z = find_z_batch(vae, model, num_samp_m=1000,
    #                     num_samp_k=10,
    #                     z=z,
    #                     max_dist=max_dist_train,
    #                     epoch=10)
    # quit()
    ######################################################################################
    # find BALD maximizing image using grad search in latent space, using handcoded ADAM #
    ######################################################################################
    z = torch.randn(2, latent_size, device=device)
    found_z = maximize_func(func=get_batch_utility_z, 
                            input_tensor=z, 
                            other_params=(vae, model))

    # Get confidence interval for batchBALD on real unseen test batch using sampled configurations/labels
    samps = torch.zeros(100)
    for i in tqdm(range(samps.shape[0])): 
        samps[i] = get_batch_utility(model, 100, 10, vae.decode(found_z[0:2]))

