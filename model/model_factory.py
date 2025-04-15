import torch 
import torchvision 
from torchvision import models



def get_model(model_name, num_class):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=False, num_classes=num_class)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    
    return model    