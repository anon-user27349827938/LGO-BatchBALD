import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from torch.autograd import Variable
import cv2
import numpy as np
import os

from params import *
from get_data import *

def _conv(channel_size, kernel_num):
    return nn.Sequential(
        nn.Conv2d(
            channel_size, kernel_num,
            kernel_size=4, stride=2, padding=1,
        ),
        nn.BatchNorm2d(kernel_num),
        nn.ReLU(),
    )

def _deconv(channel_num, kernel_num):
    return nn.Sequential(
        nn.ConvTranspose2d(
            channel_num, kernel_num,
            kernel_size=4, stride=2, padding=1,
        ),
        nn.BatchNorm2d(kernel_num),
        nn.ReLU(),
    )

def _linear(in_size, out_size, relu=True):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU(),
        ) if relu else nn.Linear(in_size, out_size)
    
    
# Define loss function
def ae_loss(x, x_hat):
    if x.shape != x_hat.shape:      # This is the case for mnist
        x = x.view(-1, 784)
    # recon_loss = F.binary_cross_entropy(x_hat, x, reduction='mean')*100
    # print(x)
    # print(x_hat)
    # quit()
    return F.mse_loss(x_hat, x, reduction='mean')
    
def train_ae(dataset_name='mnist', start_from_prev=False):
    # Load MNIST dataset
    train_loader, test_loader = get_train_test_loader(dataset_name=dataset_name, 
                                                    train_batch_size=gen_train_batch_size,
                                                    test_batch_size=gen_test_batch_size)

    # Initialize VAE model and optimizer
    if start_from_prev:
        ae = load_ae(dataset_name).to(device)
    else:
        ae = AE(dataset_name).to(device)

    optimizer = optim.Adam(ae.parameters(), 
                           lr=gen_train_learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.2,
                                                         patience=20,
                                                         min_lr=5e-5,
                                                         verbose=True)
    
    ae.train()
    print('training starts')
    for epoch in range(gen_train_num_epochs):
        for i, (images, _) in enumerate(train_loader):
            # Move images to device
            images = images.to(device)
            
            # Zero out gradients
            optimizer.zero_grad()
            
            # Forward pass
            x_hat = ae(images)
            
            # Compute loss
            loss = ae_loss(images, x_hat)

            if torch.isnan(loss):
                print('Model training stopps due to Nan loss issue!')
                torch.save(ae.state_dict(), '{}_ae.pth'.format(dataset_name))
                return
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Print progress
            if (i) % 100 == 0:
                print(f'Epoch [{epoch+1}/{gen_train_num_epochs}], \
                      Step [{i+1}/{len(train_loader)}], \
                      Loss: {loss.item():.4f}')
            scheduler.step(loss)

    torch.save(ae.state_dict(), '{}_ae.pth'.format(dataset_name))

def load_ae(dataset_name='mnist'):
    ae = AE(dataset_name)
    ae.load_state_dict(torch.load('{}_ae.pth'.format(dataset_name)))
    return ae.to(device)
     
    
class AE(nn.Module):
    def __init__(self, dataset_name='mnist'):
        super(AE, self).__init__()
        self.dataset_name = dataset_name
        self.latent_size = latent_size
        if dataset_name=='mnist':
            self.encoder = nn.Sequential(
                nn.Linear(784, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256,  latent_size) # 2 * latent_size because we need to output both mean and standard deviation
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_size, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 784),
                nn.Sigmoid()
            )
       
    def q(self, encoded):
        unrolled = encoded.view(-1, self.feature_volume)
        return self.q_mean(unrolled)
    
    def encode(self, x):
        if self.dataset_name == 'mnist':
            x = x.view(-1, 784)
            h = self.encoder(x)
            return h
    
    @torch.enable_grad()
    def decode(self, z):
        if self.dataset_name == 'mnist':
            return self.decoder(z)
        
    def forward(self, x):
        h = self.encode(x)
        return self.decode(h)


# Define VAE model
class VAE(nn.Module):
    def __init__(self, dataset_name='mnist'):
        super(VAE, self).__init__()
        self.dataset_name = dataset_name
        self.latent_size = latent_size
        if dataset_name=='mnist':
            self.encoder = nn.Sequential(
                nn.Linear(784, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 2 * latent_size) # 2 * latent_size because we need to output both mean and standard deviation
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_size, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 784),
                nn.Sigmoid()
            )
        elif dataset_name=='cifar':
            # Copied from https://github.com/SashaMalysheva/Pytorch-VAE/blob/master/model.py
            self.channel_num = 3
            self.kernel_num = 128
            self.image_size = 32

            # self.encoder = nn.Sequential(
            #     _conv(self.channel_num, self.kernel_num // 4),
            #     _conv(self.kernel_num // 4, self.kernel_num // 2),
            #     _conv(self.kernel_num // 2, self.kernel_num),
            # )
            
            # self.decoder = nn.Sequential(
            #     _deconv(self.kernel_num, self.kernel_num // 2),
            #     _deconv(self.kernel_num // 2, self.kernel_num // 4),
            #     _deconv(self.kernel_num // 4, self.channel_num),
            #     nn.Sigmoid()
            # )
            
            # encoded feature's size and volume
            self.feature_size = self.image_size // 8
            self.feature_volume = self.kernel_num * (self.feature_size ** 2)
            print('original cifar feature_vol = ', self.feature_volume)

            # q
            self.q_mean = _linear(self.feature_volume, latent_size, relu=False)
            self.q_logvar = _linear(self.feature_volume, latent_size, relu=False)

            # projection
            self.project = _linear(latent_size, self.feature_volume, relu=False)   

            # New version: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html
            act_fn = nn.ReLU
            c_hid = 32
            num_input_channels = 3
            self.encoder = nn.Sequential(
                    nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
                    nn.InstanceNorm2d(c_hid),
                    act_fn(),
                    nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(c_hid),
                    act_fn(),
                    nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
                    nn.InstanceNorm2d(2*c_hid),
                    act_fn(),
                    nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(2*c_hid),
                    act_fn(),
                    nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
                    nn.InstanceNorm2d(2*c_hid),
                    act_fn(),
                    # nn.Flatten(), # Image grid to single feature vector
                    # nn.Linear(2*16*c_hid, latent_dim)
                ) 
            
            self.decoder =  nn.Sequential(
                    nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 => 8x8
                    nn.InstanceNorm2d(2*c_hid),
                    act_fn(),
                    nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(2*c_hid),
                    act_fn(),
                    nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
                    nn.InstanceNorm2d(c_hid),
                    act_fn(),
                    nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(c_hid),
                    act_fn(),
                    nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
                    nn.Sigmoid() 
                )
            
            self.feature_volume = 64*4*4
            self.feature_size = 4
            self.kernel_num = 64
            # q
            self.q_mean = _linear(self.feature_volume, latent_size, relu=False)
            self.q_logvar = _linear(self.feature_volume, latent_size, relu=False)

            # projection
            self.project = _linear(latent_size, self.feature_volume, relu=False)   
            
       
    def q(self, encoded):
        unrolled = encoded.view(-1, self.feature_volume)
        return self.q_mean(unrolled), self.q_logvar(unrolled)
    
    def encode(self, x):
        if self.dataset_name == 'mnist':
            x = x.view(-1, 784)
            h = self.encoder(x)
            mu, log_var = torch.chunk(h, 2, dim=1)
            return mu, log_var
        elif self.dataset_name == 'cifar':
            code = self.encoder(x)
            return self.q(code)
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    @torch.enable_grad()
    def decode(self, z):
        if self.dataset_name == 'mnist':
            return self.decoder(z)
        elif self.dataset_name == 'cifar':
            z_projected = self.project(z).view(
                        -1, self.kernel_num,
                        self.feature_size,
                        self.feature_size,
                    )
            return self.decoder(z_projected)
        
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def sample(self, num_samples=1):
        """
        The function that samples images
        """
        if self.dataset_name == 'cifar':
            z = torch.randn(num_samples, latent_size).to(device)
            return self.decode(z).data

# Define loss function
def vae_loss(x, x_hat, mu, log_var):
    if x.shape != x_hat.shape:      # This is the case for mnist
        x = x.view(-1, 784)
    # recon_loss = F.binary_cross_entropy(x_hat, x, reduction='mean')*100
    # print(x)
    # print(x_hat)
    # quit()
    recon_loss = F.mse_loss(x_hat, x, reduction='mean')*10
    kl_div = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_div, recon_loss, kl_div

def train_vae(dataset_name='mnist', start_from_prev=False):
    # Load MNIST dataset
    train_loader, test_loader = get_train_test_loader(dataset_name=dataset_name, 
                                                    train_batch_size=gen_train_batch_size,
                                                    test_batch_size=gen_test_batch_size)

    # Initialize VAE model and optimizer
    if start_from_prev:
        vae = load_vae(dataset_name).to(device)
    else:
        vae = VAE(dataset_name).to(device)

    optimizer = optim.Adam(vae.parameters(), 
                           lr=gen_train_learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.2,
                                                         patience=20,
                                                         min_lr=5e-5,
                                                         verbose=True)
    
    vae.train()
    print('training starts')
    for epoch in range(gen_train_num_epochs):
        for i, (images, _) in enumerate(train_loader):
            # Move images to device
            images = images.to(device)
            
            # Zero out gradients
            optimizer.zero_grad()
            
            # Forward pass
            x_hat, mu, log_var = vae(images)
            
            # Compute loss
            loss, recon, kl = vae_loss(images, x_hat, mu, log_var)

            if torch.isnan(loss):
                print('Model training stopps due to Nan loss issue!')
                torch.save(vae.state_dict(), '{}_vae.pth'.format(dataset_name))
                return
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Print progress
            if (i) % 100 == 0:
                print(f'Epoch [{epoch+1}/{gen_train_num_epochs}], \
                      Step [{i+1}/{len(train_loader)}], \
                      Loss: {loss.item():.4f}, \
                      Recon Loss: {recon.item():.4f}, \
                      KL loss : {kl.item():.4f}')
            scheduler.step(loss)

    torch.save(vae.state_dict(), '{}_vae.pth'.format(dataset_name))

def load_vae(dataset_name='mnist'):
    vae = VAE(dataset_name)
    vae.load_state_dict(torch.load('{}_vae.pth'.format(dataset_name)))
    return vae.to(device)

def get_vae(dataset_name='mnist', force_retrain=False):
# If we can load the vae model, just load that
    save_ckpt_file = '{}_vae.pth'.format(dataset_name)
    if os.path.exists(save_ckpt_file) and not force_retrain:
        model = VAE(dataset_name).to(device)
        model.load_state_dict(torch.load(save_ckpt_file))
        return load_vae(dataset_name)
    else:
        train_vae(dataset_name, False)
        return load_vae(dataset_name)

if __name__ == '__main__':
    train(dataset_name='cifar', start_from_prev=True)
    # train(dataset_name='cifar', start_from_prev=False)
    # train(dataset_name='mnist')