import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from utils import config
def calc_cls_idx(cls_samples,f,mode):
    nc = len(cls_samples)
    
    if mode=='ca': 
        n_samples= sum(cls_samples)       
        ca_frame_num=[int((f-2)*nc*r/n_samples)+1 for r in cls_samples]
        over_flow=nc*(f-1)-sum(ca_frame_num)
        for i in range(over_flow):
            ca_frame_num[i]+=1
        ca_frame_num.reverse()
        print(ca_frame_num)
    elif mode=='ave':
        ca_frame_num = [f-1 for r in range(nc)]      
    return [sum(ca_frame_num[0:k]) for k in range(nc+1)],ca_frame_num

class CAL_classifier(nn.Module):
    def __init__( self, feat_in, num_classes, f,W=None): 
        super(CAL_classifier, self).__init__()
        W_aug = None
        if W is None:
            W_aug=nn.Parameter(nn.init.orthogonal_(torch.empty((f*num_classes,feat_in)),gain=1).data)
            W = W_aug[-num_classes:]
            scale= torch.ones((num_classes))
        self.W = nn.Parameter(W)
        self.W_aug = W_aug        
        self.scale = nn.Parameter(scale)
        self.W.requires_grad=False
        self.W_aug.requires_grad=False
        self.BN=nn.BatchNorm1d(feat_in)
    def forward(self, input):
        input = self.BN(input)
        input = input / torch.clamp(
            torch.sqrt(torch.sum(input ** 2, dim=1, keepdims=True)), 1e-8)
        return input.matmul(self.W.t())
        
class ClassAwareLoss(nn.Module):
    def __init__( self, cls_samples, in_dim, f, theta,scale=None, W_aug=None ):
        super(ClassAwareLoss, self).__init__()
        nc = len(cls_samples)
        cls_frame_idx,ca_frame_num = calc_cls_idx(cls_samples,f,'ca')
        if scale is None:
            scale= torch.ones((nc))
        self.scale = nn.Parameter(scale)
        if W_aug is None :
            W_aug=nn.init.orthogonal_(torch.empty((f*nc,in_dim)),gain=1).data
        W = W_aug[-nc:]
        V = torch.zeros_like(W_aug[:-nc,:])                
        ca_frame_denom = [np.sqrt(i) for i in ca_frame_num]
        ca_frame_nom = [np.sqrt((np.cos(theta))**2 + i*(np.sin(theta))**2) for i in ca_frame_num]
        cosine = [b/a for a,b in zip (ca_frame_denom,ca_frame_nom)]
        for i in range(nc):
            for j in range(ca_frame_num[i]):
                V[cls_frame_idx[i]+j]=np.cos(theta)*W_aug[cls_frame_idx[i]+j]+ (np.sin(theta))*W[i]       
        self.frames = nn.Parameter(V)
        self.W = nn.Parameter(W)
        self.frames.requires_grad = False
        self.W.requires_grad = False
        self.ca_frame_nom = ca_frame_nom
        self.ca_frame_denom = ca_frame_denom
        self.ca_frame_cosine = cosine    
        self.ca_frame_num = ca_frame_num # number of frames in each class
        self.ca_frame_idx = cls_frame_idx    
        self.nc=nc
        
    def forward(self, input: Tensor, target: Tensor) -> Tensor: 
        sq_norm = torch.sum(input ** 2, dim=1, keepdims=True)
        norm = torch.clamp(
            torch.sqrt(sq_norm), 1e-8)
        norm_input = input / norm
        caloss = 0
        count = 0
        reg = 0
        for index,l in enumerate(target):
            l=l.item()
            cross=0
            for j in range(self.ca_frame_num[l]):    
                cross += self.ca_frame_cosine[l]*(1-torch.dot(norm_input[index],self.frames[self.ca_frame_idx[l]+j]))**2
            count+=1                        
            reg += (norm[index]-1)**2
            caloss+=cross
        return (caloss +6e-4*reg)/count



def reg_ETF(output, label, classifier, mse_loss):
#    cur_M = classifier.cur_M
    target = classifier.cur_M[:, label].T  ## B, d
    loss = mse_loss(output, target)
    return loss

def dot_loss(output, label, cur_M, classifier, criterion, H_length, reg_lam=0):
    target = cur_M[:, label].T ## B, d  output: B, d
    if criterion == 'dot_loss':
        loss = - torch.bmm(output.unsqueeze(1), target.unsqueeze(2)).view(-1).mean()
    elif criterion == 'reg_dot_loss':
        dot = torch.bmm(output.unsqueeze(1), target.unsqueeze(2)).view(-1) #+ classifier.module.bias[label].view(-1)

        with torch.no_grad():
            M_length = torch.sqrt(torch.sum(target ** 2, dim=1, keepdims=False))
        loss = (1/2) * torch.mean(((dot-(M_length * H_length)) ** 2) / H_length)

        if reg_lam > 0:
            reg_Eh_l2 = torch.mean(torch.sqrt(torch.sum(output ** 2, dim=1, keepdims=True)))
            loss = loss + reg_Eh_l2*reg_lam

    return loss


def produce_Ew(label, num_classes):
    uni_label, count = torch.unique(label, return_counts=True)
    batch_size = label.size(0)
    uni_label_num = uni_label.size(0)
    assert batch_size == torch.sum(count)
    gamma = batch_size / uni_label_num
    Ew = torch.ones(1, num_classes).cuda(label.device)
    for i in range(uni_label_num):
        label_id = uni_label[i]
        label_count = count[i]
        length = torch.sqrt(gamma / label_count)
#        length = (gamma / label_count)
        #length = torch.sqrt(label_count / gamma)
        Ew[0, label_id] = length
    return Ew
def produce_global_Ew(cls_num_list):
    num_classes = len(cls_num_list)
    cls_num_list = torch.tensor(cls_num_list).cuda()
    total_num = torch.sum(cls_num_list)
    gamma = total_num / num_classes
    Ew = torch.sqrt(gamma / cls_num_list)
    Ew = Ew.unsqueeze(0)
    return Ew

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
