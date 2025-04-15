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
        
        
# class Lora(nn.Module):
#     def __init__(self, model, module=None, rank=8, alpha=16):
#         super().__init__()
#         self.model = model
#         self.rank = rank
#         self.alpha = alpha
#         self.hooks = []
#         self._apply_lora(module)
#         self._register_hook()
        
#     def _register_hook(self,):
#         for module in self.model.modules():        
#             def hook(module, inputs, kwargs):
#                 if 'pseudo_index' not in kwargs and hasattr(self, '_current_pseudo_index'):
#                     kwargs['pseudo_index'] = self._current_pseudo_index
#                 return inputs, kwargs 
#             hook_handle = module.register_forward_pre_hook(hook, with_kwargs=True)
#             self.hooks.append(hook_handle)
            
#     def _wrapper_forward(self, module):
#         original_forward = module.forward

#         def wrapped_forward(*args, **kwargs):
#             pseudo_label = getattr(local_storage, 'pseudo_label', None)
#             sig = inspect.signature(original_forward)
#             params = sig.parameters

#             # 如果原forward接受pseudo_label或**kwargs，则传递
#             if 'pseudo_label' in params:
#                 kwargs['pseudo_label'] = pseudo_label
#             else:
#                 # 检查是否有**kwargs
#                 has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
#                 if has_kwargs and pseudo_label is not None:
#                     kwargs['pseudo_label'] = pseudo_label

#             return original_forward(*args, **kwargs)

#         # 绑定新方法到模块
#         module.forward = wrapped_forward.__get__(module, type(module))
#         self.wrapped_modules.append(module)
        
                
#     def remove_hooks(self):
#         for hook in self.hooks:
#             hook.remove()
#         self.hooks.clear()
        
        
#     def enable(self):
#         for module in self.model.modules():
#             if isinstance(module, LoraLinear):
#                 module.base_layer.requires_grad = False
#                 module.lora_A.requires_grad = True
#                 module.lora_B.requires_grad = True

#     def _apply_lora(self, module=None):
#         if module is None:
#             module = self.model
            
#         for name, child in module.named_children():
#             if isinstance(child, nn.Linear):
#                 new_layer = LoraLinear(child, self.rank, self.alpha)
#                 setattr(module, name, new_layer)
#             else:
#                 self._apply_lora(child)

    
#     def forward(self, x, *args, **kwargs):
#         """
#         Forward pass through the model with LoRA adjustments.
#         """
#         # if 'pseudo_index' not in kwargs:
#         #     kwargs['pseudo_index'] = None
#         #     with torch.no_grad():
#         #         pseduo_label = self.model(x, *args, **kwargs)
#         #     return self.model(x, pseudo_index=pseduo_label, *args, **kwargs)
#         # else:
#         #     return self.model(x, pseudo_index=pseduo_label, *args, **kwargs)

#         if 'pseudo_index' not in kwargs:
#             with torch.no_grad():
#                 pseudo_label = self.model(x, *args, **kwargs)
#             print('pseudo_label', pseudo_label)
#             # 保存伪标签到类里或其它地方（比如 self.pseudo_index），不要再传给整个模型
#             kwargs['pseudo_index'] = pseudo_label
#         return self.model(x, *args, **kwargs)

class CALora(nn.Module):
    def __init__(self, model, rank=8, alpha=16):
        super().__init__()
        self.model = model  
        
        self.linear = model.fc
        self.rank = rank
        self.alpha = alpha
        self.hooks = []
        
        self._apply_calora()
        
        self._register_hook()
        
        self._freeze_other_parameters()
        
    
        
        
        
    def _freeze_other_parameters(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self.linear._enable_lora()
        
        
    
    def _apply_calora(self):
        # module = model.fc
        self.linear = CALoraLinear(self.linear, self.model.fc.in_features, self.rank, self.alpha)

    
    def _get_pseudo_label(self, x):
        with torch.no_grad():
            pseudo_label = self.model(x)
        return pseudo_label 
    
    def _register_hook(self):
        def hook_fn(module, input, output):
            self.fc_input = input[0]  # 捕捉 fc 层的输入
        
        handle = self.model.fc.register_forward_hook(hook_fn)
        self.hooks.append(handle)
    
    def forward(self, x):
        pseudo_label = self._get_pseudo_label(x)
        return self.linear(self.fc_input, pseudo_index=pseudo_label)
        
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
    

        
    


if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)
    model = models.resnet18(pretrained=False, num_classes=10)
    lora_model = CALora(model, rank=8, alpha=16)
    print(lora_model)
    pseudo_label = lora_model._get_pseudo_label(x)
    predict = lora_model(x)
    print(pseudo_label.shape, pseudo_label)
    print(predict.shape, predict)