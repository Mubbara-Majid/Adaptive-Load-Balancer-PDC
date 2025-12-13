import torch
from torchvision import datasets, transforms
import os

def get_data_loader(batch_size=64):
    if not os.path.exists('./data'):
        os.makedirs('./data')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return loader