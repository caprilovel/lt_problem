import torch
import torch.nn as nn
from functools import partial
from torch.nn import Parameter
import math

# class LoraLinear(nn.Module):
#     def __init__(self, base_layer, rank=8, alpha=16):
#         super().__init__()
#         self.base_layer = base_layer
        
#         # Freeze original parameters
#         for param in base_layer.parameters():
#             param.requires_grad = False

#         # Add LoRA parameters
#         in_features = base_layer.in_features
#         out_features = base_layer.out_features
        
#         self.lora_A = Parameter(torch.empty((rank, in_features)))
#         self.lora_B = Parameter(torch.zeros((out_features, rank)))
#         self.scaling = alpha / rank
        
#         # Initialize LoRA weights
#         nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
#         self.enabled = True  
#         # Preserve original layer's attributes
#         self.bias = base_layer.bias
        
          
#     def forward(self, x, pseudo_index=None):
#         if torch.is_grad_enabled() or self.enabled:  # if run in the train mode second round or in the inference mode 
#             orig_output = self.base_layer(x)  
#             lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling  # LoRA计算
#             return orig_output + lora_output + (self.bias if self.bias is not None else 0)
#         else:  # if run with the no_grad, then only run the base layer
#             return self.base_layer(x)
class LoraLinear(nn.Module):
    def __init__(self, base_layer, num_class, rank=8, alpha=16, top_k=1):
        super().__init__()
        self.base_layer = base_layer
        self.num_class = num_class
        self.rank = rank
        self.alpha = alpha
        self.top_k = top_k
        
        
        # Freeze original parameters
        for param in base_layer.parameters():
            param.requires_grad = False

        # Add LoRA parameters
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        self.lora_A = nn.Parameter(torch.empty((num_class * rank, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, num_class * rank)))
        self.scaling = alpha / rank
        
        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        # Preserve original layer's attributes
        self.bias = base_layer.bias
        
        
    def _topk_lora(self, pseduo_index):
        '''
        Args:
            pseduo_index: [batch_size, num_class]
        '''
        topk_value, topk_index = torch.topk(pseduo_index, self.top_k, dim=1)
        # duplicate from batch_size, num_class to batch_size, num_class * rank
        top_index = topk_index.unsqueeze(2).expand(-1, -1, self.rank)  # [batch_size, num_class, rank]
        top_index = top_index.reshape(-1)  # [batch_size * num_class * rank]
        
        return top_index
        

    def forward(self, x, pseudo_index=None):
        if pseudo_index == None:  # in the first round, only run the base layer and get the pseudo index 
            return self.base_layer(x)
            
        else:  # if run with the no_grad, then only run the base layer
            orig_output = self.base_layer(x)  
            top_index = self._topk_lora(pseudo_index)
            
            # Retrieve and reshape LoRA parameters
            A = self.lora_A[top_index]  # (batch_size * top_k * rank, in_features)
            A = A.view(-1, self.top_k * self.rank, A.size(-1))  # (batch_size, top_k * rank, in_features)
            
            B = self.lora_B[:, top_index]  # (out_features, batch_size * top_k * rank)
            B = B.view(B.size(0), -1, self.top_k * self.rank)  # (out_features, batch_size, top_k * rank)
            B = B.permute(1, 0, 2)  # (batch_size, out_features, top_k * rank)
            
            # Compute delta weights for each sample in the batch
            delta_W = torch.bmm(B, A)  # (batch_size, out_features, in_features)
            
            # Apply LoRA adjustments
            lora_output = (x.unsqueeze(1) @ delta_W.transpose(1, 2)).squeeze(1) * self.scaling
            final_output = orig_output + lora_output
            return final_output + (self.bias if self.bias is not None else 0)
        
    def _update_from_linear(self, linear_layer):
        """
        Convert a standard linear layer to a LoRA linear layer.
        """
        self.base_layer.weight = linear_layer.weight
        self.base_layer.bias = linear_layer.bias
        self.base_layer.in_features = linear_layer.in_features
        self.base_layer.out_features = linear_layer.out_features



