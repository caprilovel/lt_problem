import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
from torch.utils.data import DataLoader, Subset

import wandb

from dataset.cifar import get_cifar_dataset
from engine import train



wandb.init(project="cifar10-lora")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataloader, test_loader = get_cifar_dataset(path='./data/')

model = models.resnet18(pretrained=False, num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)





train(model, dataloader, criterion, optimizer, test_loader, device, epochs=30)
