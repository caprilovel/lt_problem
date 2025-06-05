import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import math
import inspect




class RouteLinear(nn.Module):
    def __init__(self, linear_layer, num_classes):
        super(RouteLinear, self).__init__()
        self.linear_layer_weight = linear_layer.weight
        self.linear_layer_weight.requires_grad = False
        self.linear_layer_bias = linear_layer.bias
        self.linear_layer_bias.requires_grad = False
        self.u = torch.nn.Parameter(torch.randn(num_classes, linear_layer.weight.size(1)))
        self.v = torch.nn.Parameter(torch.randn(num_classes, linear_layer.weight.size(0)))
        
        self.attention_route = nn.Linear(linear_layer.weight.size(1), num_classes)
        
        
    def forward(self, x):
        attention = torch.sigmoid(self.attention_route(x))
        # choose top 1 class of the attention
        lora_weight = torch.matmul(attention, self.u).T @ torch.matmul(self.v, self.linear_layer_weight)
        return F.linear(x, lora_weight + self.linear_layer_weight, self.linear_layer_bias)

    
class LoRAAttentionRouter(nn.Module):
    def __init__(self, in_features, num_classes, k=1):
        super(LoRAAttentionRouter, self).__init__()
        self.k = k
        self.attention = nn.Linear(in_features, num_classes)
        
        self.lora_ranks = nn.Parameter(torch.randn(num_classes, in_features))
        
    def forward(self, features):
        
        attn_scores = self.attention(features)  # [B, num_classes]
        
        
        topk_values, topk_indices = torch.topk(
            attn_scores, k=self.k, dim=1
        )  # [B, k]
        
        
        topk_weights = torch.softmax(topk_values, dim=1)  # [B, k]
        
        
        selected_loras = self.lora_ranks[topk_indices]  # [B, k, in_features]
        
        
        combined_lora = torch.einsum(
            "bk,bki->bi", topk_weights, selected_loras
        )  # [B, in_features]
        
        
        adapted_features = features + combined_lora
        return adapted_features

class LoRAResNet18(nn.Module):
    def __init__(self, num_classes=10, base_model=models.resnet18(pretrained=False), k=1):
        super(LoRAResNet18, self).__init__()
        self.resnet = base_model
        in_features = self.resnet.fc.in_features
        
        
        self.resnet.fc = nn.Identity()
        
        
        self.router = LoRAAttentionRouter(
            in_features=in_features,
            num_classes=num_classes,
            k=k
        )
        
        
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        
        features = self.resnet(x)
        features = torch.flatten(features, 1)
        
        
        adapted_features = self.router(features)
        
        
        out = self.fc(adapted_features)
        return out


class CALoraLinear(nn.Module):
    def __init__(self, base_layer, num_class, rank=1, alpha=16, top_k=4):
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
        
    def _enable_lora(self,):
        self.base_layer.requires_grad = False
        self.lora_A.requires_grad = True
        self.lora_B.requires_grad = True
        
        
        self.enabled = True  # Set a flag to indicate LoRA is enabled

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

class UCALoraLinear(nn.Module):
    def __init__(self, base_layer, num_class, rank=10, alpha=16, top_k=3):
        super().__init__()
        self.base_layer = base_layer
        self.num_class = num_class
        self.rank = rank
        self.alpha = alpha
        self.top_k = top_k
        
        self.class_mask = torch.ones(num_class, rank)
        # first five class only keep half of the rank
        self.class_mask[:7,] = 0
        
        # self.class_mask = self.class_mask.view(-1)
        self.class_mask = self.class_mask.to(base_layer.weight.device)
        # print(self.class_mask)
        
        # Freeze original parameters
        # for param in base_layer.parameters():
        #     param.requires_grad = False

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
            
        Returans:
            top_index: [batch_size * num_class * rank]
        '''
        topk_value, topk_index = torch.topk(pseduo_index, self.top_k, dim=1)
        # duplicate from batch_size, num_class to batch_size, num_class * rank
        mask = torch.zeros_like(pseduo_index)
        mask.scatter_(1, topk_index, 1)
        # expand mask to rank
        mask = mask.unsqueeze(2).expand(-1, -1, self.rank)  # [batch_size, num_class, rank]
        # print('mask', mask.shape, self.class_mask.shape)
        # mask = mask * self.class_mask.unsqueeze(0)  # [batch_size, num_class, rank]
        # print('mask', mask[0])
        mask = mask.reshape(mask.shape[0], -1)  # [batch_size, num_class * rank]
                         
        # change mask into index back
        top_index = mask.nonzero(as_tuple=True)[1]  # [batch_size, num_class * rank]
        
        top_index = top_index.reshape(-1)  # [batch_size * num_class * rank]
        
        return top_index
        
    def _topk_lora_mask(self, pseduo_index):
        '''
        Args:
            pseduo_index: [batch_size, num_class]
            
        Returans:
            top_index: [batch_size * num_class * rank]
        '''
        topk_value, topk_index = torch.topk(pseduo_index, self.top_k, dim=1)
        # duplicate from batch_size, num_class to batch_size, num_class * rank
        mask = torch.zeros_like(pseduo_index)
        mask.scatter_(1, topk_index, 1)
        # expand mask to rank
        mask = mask.unsqueeze(2).expand(-1, -1, self.rank)  # [batch_size, num_class, rank]
        # print('mask', mask.shape, self.class_mask.shape)
        mask = mask * self.class_mask.unsqueeze(0)  # [batch_size, num_class, rank]
        # print('mask', mask[0])
        mask = mask.view(-1, self.num_class * self.rank)  # [batch_size, num_class * rank]
        # top_index = topk_index.unsqueeze(2).expand(-1, -1, self.rank)  # [batch_size, num_class, rank]
        # change mask into index back
        return mask
    
    def _enable_lora(self,):
        self.base_layer.requires_grad = False
        self.lora_A.requires_grad = True
        self.lora_B.requires_grad = True
        
        
        self.enabled = True  # Set a flag to indicate LoRA is enabled

    def forward(self, x, pseudo_index=None):
        if pseudo_index == None:  # in the first round, only run the base layer and get the pseudo index 
            return self.base_layer(x)
            
        else:  # if run with the no_grad, then only run the base layer
            orig_output = self.base_layer(x)  
            top_index_mask = self._topk_lora_mask(pseudo_index)
            
            # Retrieve and reshape LoRA parameters
            
            mask_A = top_index_mask.unsqueeze(2)
            A = self.lora_A.unsqueeze(0) * mask_A  # (batch_size * top_k * rank, in_features)
            
            mask_B = top_index_mask.unsqueeze(1)
            B = self.lora_B.unsqueeze(0) * mask_B  # (out_features, batch_size * top_k * rank)

            delta_W = torch.bmm(B, A)  # (batch_size, out_features, in_features)
            
            # Apply LoRA adjustments
            lora_output = (x.unsqueeze(1) @ delta_W.transpose(1, 2)).squeeze(1) * self.scaling
            final_output = orig_output + lora_output
            return final_output + (self.bias if self.bias is not None else 0)
        
    def _forward(self, x, pseudo_index=None):
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
            # calculate the mask
            
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
                


# class CALora(nn.Module):
#     def __init__(self, model, rank=2, alpha=16, topk=4, num_class=10):
#         super().__init__()
#         self.model = model  
        
#         self.linear = model.fc
#         self.rank = rank
#         self.alpha = alpha
#         self.topk = topk
#         self.num_class = num_class
#         self.hooks = []
        
#         self._apply_calora()
        
#         self._register_hook()
        
#         self._freeze_other_parameters()
        
    
        
        
        
#     def _freeze_other_parameters(self):
#         for param in self.model.parameters():
#             param.requires_grad = False
#         self.linear._enable_lora()
        
        
    
#     def _apply_calora(self):
#         # module = model.fc
#         self.linear = UCALoraLinear(
#             self.linear,
#             self.num_class,
#             self.rank, 
#             self.alpha,
#             self.topk
#             )

    
#     def _get_pseudo_label(self, x):
#         pseudo_label = self.model(x)
#         return pseudo_label 
    
#     def _register_hook(self):
#         def hook_fn(module, input, output):
#             self.fc_input = input[0]
        
#         handle = self.model.fc.register_forward_hook(hook_fn)
#         self.hooks.append(handle)
    
#     def forward(self, x):
#         pseudo_label = self._get_pseudo_label(x)
#         return self.linear(self.fc_input, pseudo_index=pseudo_label)
        
#     def remove_hooks(self):
#         for hook in self.hooks:
#             hook.remove()
#         self.hooks.clear()
    
#     def get_features(self, x):
#         features = self.model.get_features(x)
#         # print(features.requires_grad)
#         return features
        
    
class CALora(nn.Module):
    def __init__(self, model, rank=2, alpha=16, topk=4, num_classes=10):
        super().__init__()
        self.model = model

        # 用 UCALoraLinear 替换 model.fc
        self.model.fc = UCALoraLinear(
            base_layer=self.model.fc,
            num_class=num_classes,
            rank=rank,
            alpha=alpha,
            top_k=topk
        )
        self.model.fc._enable_lora()
        self.fc = self.model.fc  # 保存对 LoRA 层的引用

    def forward(self, x, pseudo_index=None):
        if pseudo_index is None:
            return self.model(x)
        else:
            features = self.model.get_features(x)
            return self.fc(features, pseudo_index=pseudo_index)
        

    def get_features(self, x):
        return self.model.get_features(x)

        
    


if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)
    model = models.resnet18(pretrained=False, num_classes=10)
    lora_model = CALora(model, rank=8, alpha=16)
    print(lora_model)
    pseudo_label = lora_model._get_pseudo_label(x)
    predict = lora_model(x)
    print(pseudo_label.shape, pseudo_label)
    print(predict.shape, predict)