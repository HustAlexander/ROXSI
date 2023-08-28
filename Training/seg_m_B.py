import torch
from torch.autograd import Function,Variable
from torchvision import models
from torchvision import utils
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import torch.nn as nn
from torch import optim
import random
import logging
import sys
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-8
        self.bce = nn.BCELoss()
    
    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)
        
        pre = predict.view(num, -1)
        tar = target.view(num, -1)
        
        intersection = (pre * tar).sum(-1).sum()  #利用预测值与标签相乘当作交集
        union = (pre + tar).sum(-1).sum()
        
        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        # score += self.bce(predict,target)
        
        return score

cos = nn.CosineSimilarity(dim=1, eps=1e-6)
criterion1 = DiceLoss()
criterion2 = nn.L1Loss()

def cos_loss(CAM1, CAM2, eps=1e-7):
    CAM1 = CAM1.view(CAM1.shape[0], -1)#将特征转换为N*(C*W*H)，即两维
    CAM2 = CAM2.view(CAM2.shape[0], -1)

    distance= cos(CAM1,CAM2).mean()
    
    
    return distance


def seg_m_B(input, Mask1,Mask2,Mask4,model,optimizer):

        
        #### grad_cam########
        output = model(input)

       

        output = torch.sigmoid(output)
        out1 = output[:,0,:].unsqueeze(1)
        out2 = output[:,1,:].unsqueeze(1)
        out4 = output[:,2,:].unsqueeze(1)




        loss1 = criterion1(out1,Mask1)+criterion1(out2,Mask2)+criterion1(out4,Mask4)
        loss2 = 0
        loss3 = 0

        optimizer.zero_grad()
        (loss1+0.1*loss2+0.1*loss3).backward()
        optimizer.step()
        
        


    
        return loss1,loss2,loss3,output



            
    

   