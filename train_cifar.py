import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
from torch.utils.data import DataLoader, Subset
import warnings
warnings.filterwarnings("ignore")

import wandb
import datetime

from dataset.cifar import get_cifar_dataset
from engine import train
from utils import parse_args, get_device, random_fix


class ResNetWithFeatures(nn.Module):
    def __init__(self, base_model, num_classes=10):
        super(ResNetWithFeatures, self).__init__()
        
        resnet = base_model

        
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # 去掉最后一层 fc
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)

    def get_features(self, x):
        x = self.backbone(x)  # [B, C, 1, 1]
        x = torch.flatten(x, 1) 
        return x

    def forward(self, x):
        feats = self.get_features(x)
        out = self.fc(feats)
        return out

if __name__ == "__main__":
    random_fix(2024)
    parser = parse_args()
    print(f"Arguments: {vars(parser)}")
    if parser.wandb:
        wandb.init(project="cifar10-lora", entity=parser.wandb_entity, config=vars(parser), name=f"{parser.model}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        print(f"Initialized wandb with run id: {wandb.run.id}")

    device = get_device(parser.device)
    dataloader, test_loader = get_cifar_dataset(path='./data/')

    model = models.resnet18(pretrained=False, num_classes=10).to(device)
    model = ResNetWithFeatures(base_model=model, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train(model, dataloader, criterion, optimizer, test_loader, device, 
          epochs=parser.n_epoch,
          parser=parser)
    

