
import numpy as np
import torch 
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms

from torchvision import datasets, models

def get_cifar_dataset(path='./data/'):
    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.CIFAR10(root=path, train=True, download=True, transform=data_transform)
    


    def create_long_tail(dataset, imbalance_ratio=0.02):
        class_counts = np.array([5000 * (imbalance_ratio ** (i / 9.0)) for i in range(10)]).astype(int)
        indices = []
        targets = np.array(dataset.targets)
        for class_idx in range(10):
            class_indices = np.where(targets == class_idx)[0]
            selected_indices = np.random.choice(class_indices, class_counts[class_idx], replace=False)
            indices.extend(selected_indices)
        return Subset(dataset, indices)


    long_tail_dataset = create_long_tail(dataset)
    dataloader = DataLoader(long_tail_dataset, batch_size=128, shuffle=True, num_workers=4)


    test_dataset = datasets.CIFAR10(root=path, train=False, download=True, transform=data_transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    return dataloader, test_loader


def get_cifar_100_dataset(path='./data/'):
    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR100(root=path, train=True, download=True, transform=data_transform)
    
    
    def create_long_tail(dataset, imbalance_ratio=0.02):
        class_counts = np.array([5000 * (imbalance_ratio ** (i / 99.0)) for i in range(100)]).astype(int)
        indices = []
        targets = np.array(dataset.targets)
        for class_idx in range(100):
            class_indices = np.where(targets == class_idx)[0]
            selected_indices = np.random.choice(class_indices, class_counts[class_idx], replace=False)
            indices.extend(selected_indices)
        return Subset(dataset, indices)
    long_tail_dataset = create_long_tail(dataset)
    dataloader = DataLoader(long_tail_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_dataset = datasets.CIFAR100(root=path, train=False, download=True, transform=data_transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    return dataloader, test_loader





