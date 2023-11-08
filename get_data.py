import torch
from torchvision.datasets import ImageFolder, MNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from params import *

def MNIST_transform():
    return transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5])
    ])



def get_full_loader(dataset_name = 'mnist', batch_size= oracle_train_batch_size):
    if dataset_name == 'mnist':
        train_dataset = MNIST(root='data/', train=True, 
                              transform=MNIST_transform(), download=True)
        test_dataset = MNIST(root='data/', train=False, 
                             transform=MNIST_transform(), download=True)
        full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        full_loader = DataLoader(dataset=full_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=True, pin_memory=(device=='gpu'))
    return full_loader


def get_train_test_loader(dataset_name, 
                          train_batch_size, 
                          test_batch_size,
                          device=device):  # Add a 'device' argument with a default value of 'cpu'
    
    if device == 'gpu':
        if not torch.cuda.is_available():
            print("Warning: GPU not available. Using CPU instead.")
            device = 'cpu'
    
    if dataset_name == 'mnist':
        train_dataset = MNIST(root='data/', train=True, 
                              transform=MNIST_transform(), download=True)
        train_loader = DataLoader(dataset=train_dataset, 
                                  batch_size=train_batch_size, 
                                  shuffle=True, pin_memory=(device=='gpu'))

        test_dataset = MNIST(root='data/', train=False, 
                             transform=MNIST_transform(), download=True)
        test_loader = DataLoader(dataset=test_dataset, 
                                 batch_size=test_batch_size, shuffle=True, pin_memory=(device=='gpu'))
        return train_loader, test_loader
    elif dataset_name == 'cifar':
        train_dataset = CIFAR10(root='data/', train=True, 
                              transform=MNIST_transform(), download=True)
        train_loader = DataLoader(dataset=train_dataset, 
                                  batch_size=train_batch_size, 
                                  shuffle=True, pin_memory=(device=='gpu'))

        test_dataset = CIFAR10(root='data/', train=False, 
                             transform=MNIST_transform(), download=True)
        test_loader = DataLoader(dataset=test_dataset, 
                                 batch_size=test_batch_size, shuffle=True, pin_memory=(device=='gpu'))
        return train_loader, test_loader
    else:
        print('ERROR! Your dataset name is not supported currently')
        quit()

