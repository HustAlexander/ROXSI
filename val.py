from config import opt
import torch
import dataset
from torch.autograd import Function,Variable
from torchvision import models
from torchvision import utils
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import torch.nn as nn
from torch import optim
from evaluation import *
import random
from model_seg import IU_Net_5_1_val,IU_Net_5_1_a_val,IU_Net_5,IU_Net_5_1_t,IU_Net_5_1_d_val,IU_Net_5_1_k,IU_Net_5_1_n

from Missformer.MISSFormer import MISSFormer
import logging
import sys
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


from medpy.metric.binary import hd,hd95



random.seed(309)
np.random.seed(309)
torch.manual_seed(309)
torch.cuda.manual_seed_all(309)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def MV(r):
    d = []
    n=500
    for i in range(3):
        y = list(r[i])
        mean = []
        se = []
        for _ in range(n):
            a =  random.sample(y, 30)
            a = np.array(a)
            mean.append(np.mean(a))
            se.append(np.std(a))
            
        b = np.array(mean).reshape(n,1)
        c = np.array(se).reshape(n,1)
        b = np.concatenate((b,c),axis=1)
        b = b[np.argsort(b[:,0])]
        lower = b[int(n*0.025)-1,:]
        higer = b[int(n*0.975),:]
        b = b[int(n*0.025)-1:int(n*0.975),:]
        b = np.concatenate((b.mean(axis=0),lower,higer),axis=0)
        d.append(b) 

    return d

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


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--training")
    parser.add_argument("--in_channels")
    parser.add_argument("--extra",type=str,default="None")
    parser.add_argument("--aug",type=str)
    parser.add_argument("--modal",type=int,default = 4)
    parser.add_argument("--fold",type=str,default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    outputs_dir = '/storage/homefs/zk22e821/MM'
    if not os.path.exists(outputs_dir):
        os.mkdir(outputs_dir)
    logs_dir = os.path.join(outputs_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)

    

    def count_param(model):
        param_count = 0
        for param in model.parameters():
            param_count += param.view(-1).size()[0]
        return param_count




    if args.extra == "5_1_a":
        model = IU_Net_5_1_a_val(feature_ch=int(args.in_channels),output_ch=3).cuda()
        state_dict = torch.load('/storage/homefs/zk22e821/MM/models/continue/'+'seg_extra_B_5_1_a_'+args.fold+'.ckpt')
        model.load_state_dict(state_dict,strict=False)
        model = model.cuda()


    elif args.extra == "5":
        model = IU_Net_5(feature_ch=int(args.in_channels),output_ch=3).cuda()
        state_dict = torch.load('/storage/homefs/zk22e821/MM-19/models/continue/'+'seg_m_B_5_'+args.fold+'.ckpt')
        model.load_state_dict(state_dict,strict=False)
        model = model.cuda()


        writer = SummaryWriter(log_dir="./runs/"+args.training+args.extra+"_"+args.fold+'_extra')



    elif args.extra == "5_1_d":
        model = IU_Net_5_1_d_val(feature_ch=int(args.in_channels),output_ch=3).cuda()

        state_dict = torch.load('/storage/homefs/zk22e821/MM/models/continue/'+'seg_extra_BV_5_1_d_'+args.fold+'.ckpt')
        model.load_state_dict(state_dict,strict=False)
        model = model.cuda()
        writer = SummaryWriter(log_dir="./runs/"+args.training+args.extra+"_"+args.fold+'_extra')


    elif args.extra == "missformer":
        model = MISSFormer(num_classes=3).cuda()

        state_dict = torch.load('/storage/homefs/zk22e821/MM/models/continue/'+'seg_m_B_missformer_'+args.fold+'.ckpt')
        model.load_state_dict(state_dict,strict=False)
        model = model.cuda()
        writer = SummaryWriter(log_dir="./runs/"+args.training+args.extra+"_"+args.fold+'_extra')

    
    
    

    # train_data = dataset.createval(opt.data_root20, train=True,fold=args.fold)
    val_data = dataset.createval(opt.data_root20, train=False,fold=args.fold, aug = args.aug, modal = args.modal)

    val_dataloader = DataLoader(val_data, 1)

    lr=opt.lr

    



        


    model.eval()
    l = 0
    total=0
    SE1,SP1,DC1,SE2,SP2,DC2,SE4,SP4,DC4 = 0.,0.,0.,0.,0.,0.,0.,0.,0.
    SE4_1,SP4_1,DC4_1,SE4_0,SP4_0,DC4_0,SE4_2,SP4_2,DC4_2,SE4_3,SP4_3,DC4_3 = 0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.
    DC4_0,DC2_0,DC1_0,DC4_1,DC2_1,DC1_1,DC4_2,DC2_2,DC1_2,DC4_3,DC2_3,DC1_3 = 0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.

    SE_WT,SP_WT,DC_WT,SE_TC,SP_TC,DC_TC = 0.,0.,0.,0.,0.,0.
    etn,wtn,tcn=0,0,0

    id = 0
    s = 1
    p = 0
    sp,se,dc=[],[],[]
    dc_w = []
    dc_t = []
    dc_e = []
    hd_et,hd_wt,hd_tc = [],[],[]

    with torch.no_grad():
        for i, (data, label1,label2,label4,Mask1,Mask2,Mask4,Mask,data_name) in enumerate(val_dataloader):
            input = Variable(data.cuda())
            Mask1 = Variable(Mask1.cuda()).float()
            Mask2 = Variable(Mask2.cuda()).float()
            Mask4 = Variable(Mask4.cuda()).float()

            Mask = Variable(Mask.cuda()).long()

            WT = Mask1+Mask2+Mask4

            TC = Mask1+Mask4

            ET = Mask4

            label1 = Variable(label1.cuda()).long()
            label2 = Variable(label2.cuda()).long()
            label4 = Variable(label4.cuda()).long()
            label = ((label1+label2+label4)>0).long()

            GT = Variable(Mask.cuda()).squeeze(1).long()

            if args.training =="seg_m_B" and args.extra =="None":
                out = model(input)
                out0,out1,out2,out3 = 0*out,0*out,0*out,0*out
            elif args.training =="seg_extra_B" and args.extra =="5_1_d":
                out  = model(input)
            else:
                out  = model(input)



            
            SR1 = torch.sigmoid(out[:,0,:])
            SR2 = torch.sigmoid(out[:,1,:])
            SR4 = torch.sigmoid(out[:,2,:])

            SR1 = SR1.unsqueeze(1)
            SR2 = SR2.unsqueeze(1)
            SR4 = SR4.unsqueeze(1)


            # out0_1 = torch.sigmoid(out0[:,0,:]).unsqueeze(1)
            # out0_2 = torch.sigmoid(out0[:,1,:]).unsqueeze(1)
            # out0_4 = torch.sigmoid(out0[:,2,:]).unsqueeze(1)


            # out1_1 = torch.sigmoid(out1[:,0,:]).unsqueeze(1)
            # out1_2 = torch.sigmoid(out1[:,1,:]).unsqueeze(1)
            # out1_4 = torch.sigmoid(out1[:,2,:]).unsqueeze(1)

            # out2_1 = torch.sigmoid(out2[:,0,:]).unsqueeze(1)
            # out2_2 = torch.sigmoid(out2[:,1,:]).unsqueeze(1)
            # out2_4 = torch.sigmoid(out2[:,2,:]).unsqueeze(1)



            # out3_1 = torch.sigmoid(out3[:,0,:]).unsqueeze(1)
            # out3_2 = torch.sigmoid(out3[:,1,:]).unsqueeze(1)
            # out3_4 = torch.sigmoid(out3[:,2,:]).unsqueeze(1)



            # loss1 = criterion1(SR1,WT)+criterion1(SR2,TC)+criterion1(SR4,ET)
            # l += loss1

            SR1 = SR1>0.5
            SR2 = SR2>0.5
            SR4 = SR4>0.5

        ####### evaluate on SCAN #########
            if data_name[0].split('_')[-3].split('/')[-1] != id:

                if p > 0:
                    if torch.sum(labelS) != 0:
                        SE1 += get_sensitivity(SR1S, WTS)
                        SP1 += get_specificity(SR1S, WTS)
                        DC1 += get_DC(SR1S, WTS)

                        SE2 += get_sensitivity(SR2S, TCS)
                        SP2 += get_specificity(SR2S, TCS)
                        DC2 += get_DC(SR2S, TCS)

                        SE4 += get_sensitivity(SR4S, ETS)
                        SP4 += get_specificity(SR4S, ETS)
                        DC4 += get_DC(SR4S, ETS)

                        # dc.append((get_DC(SR1S, WTS)+get_DC(SR2S, TCS)+get_DC(SR4S, ETS))/3)
                        dc_w.append(get_DC(SR1S, WTS))
                        dc_t.append(get_DC(SR2S, TCS))  
                        dc_e.append(get_DC(SR4S, ETS))         
                        if SR4S.sum()>0 and ETS.sum()>0:
                            etn+=1
                            hd_et.append(hd(SR4S.cpu().numpy(), ETS.cpu().numpy()))
                        if SR2S.sum()>0 and TCS.sum()>0:
                            tcn+=1
                            hd_tc.append(hd(SR2S.cpu().numpy(), TCS.cpu().numpy()))
                        if SR1S.sum()>0 and WTS.sum()>0:
                            wtn+=1
                            hd_wt.append(hd95(SR1S.cpu().numpy(), WTS.cpu().numpy()))

                        # DC4_0 += get_DC(SR4_0S, ETS)
                        # DC2_0 += get_DC(SR2_0S, TCS)
                        # DC1_0 += get_DC(SR1_0S, WTS)

                        # DC4_1 += get_DC(SR4_1S, ETS)
                        # DC2_1 += get_DC(SR2_1S, TCS)
                        # DC1_1 += get_DC(SR1_1S, WTS)


                        # DC4_2 += get_DC(SR4_2S, ETS)
                        # DC2_2 += get_DC(SR2_2S, TCS)
                        # DC1_2 += get_DC(SR1_2S, WTS)


                        # DC4_3 += get_DC(SR4_3S, ETS)
                        # DC2_3 += get_DC(SR2_3S, TCS)
                        # DC1_3 += get_DC(SR1_3S, WTS)
                        


                id = data_name[0].split('_')[-3].split('/')[-1]
                SR1S,WTS,SR2S,TCS,SR4S,ETS = SR1,WT,SR2,TC,SR4,ET
                # SR4_0S,SR4_1S,SR4_2S,SR4_3S = out0_4,out1_4,out2_4,out3_4
                # SR2_0S,SR2_1S,SR2_2S,SR2_3S = out0_2,out1_2,out2_2,out3_2
                # SR1_0S,SR1_1S,SR1_2S,SR1_3S = out0_1,out1_1,out2_1,out3_1

                labelS = label
                s = 1
                p += 1
            else:
                SR1S = torch.cat((SR1S,SR1),1)
                WTS = torch.cat((WTS,WT),1)
                SR2S = torch.cat((SR2S,SR2),1)
                TCS = torch.cat((TCS,TC),1)
                SR4S = torch.cat((SR4S,SR4),1)
                ETS = torch.cat((ETS,ET),1)

                # SR4_0S = torch.cat((SR4_0S,out0_4),1)
                # SR4_1S = torch.cat((SR4_1S,out1_4),1)
                # SR4_2S = torch.cat((SR4_2S,out2_4),1)
                # SR4_3S = torch.cat((SR4_3S,out3_4),1)
                # SR2_0S = torch.cat((SR2_0S,out0_2),1)
                # SR2_1S = torch.cat((SR2_1S,out1_2),1)
                # SR2_2S = torch.cat((SR2_2S,out2_2),1)
                # SR2_3S = torch.cat((SR2_3S,out3_2),1)
                # SR1_0S = torch.cat((SR1_0S,out0_1),1)
                # SR1_1S = torch.cat((SR1_1S,out1_1),1)
                # SR1_2S = torch.cat((SR1_2S,out2_1),1)
                # SR1_3S = torch.cat((SR1_3S,out3_1),1)
                labelS += label
                s += 1

        ###last scan###  
        if torch.sum(labelS) != 0:
            SE1 += get_sensitivity(SR1S, WTS)
            SP1 += get_specificity(SR1S, WTS)
            DC1 += get_DC(SR1S, WTS)

            SE2 += get_sensitivity(SR2S, TCS)
            SP2 += get_specificity(SR2S, TCS)
            DC2 += get_DC(SR2S, TCS)

            SE4 += get_sensitivity(SR4S, ETS)
            SP4 += get_specificity(SR4S, ETS)
            DC4 += get_DC(SR4S, ETS)
            # dc.append((get_DC(SR1S, WTS)+get_DC(SR2S, TCS)+get_DC(SR4S, ETS))/3)
            dc_w.append(get_DC(SR1S, WTS))
            dc_t.append(get_DC(SR2S, TCS))  
            dc_e.append(get_DC(SR4S, ETS)) 
            if SR4S.sum()>0 and ETS.sum()>0:
                etn+=1
                hd_et.append(hd(SR4S.cpu().numpy(), ETS.cpu().numpy()))
            if SR2S.sum()>0 and TCS.sum()>0:
                tcn+=1
                hd_tc.append(hd(SR2S.cpu().numpy(), TCS.cpu().numpy()))
            if SR1S.sum()>0 and WTS.sum()>0:
                wtn+=1
                hd_wt.append(hd95(SR1S.cpu().numpy(), WTS.cpu().numpy()))


            # DC4_0 += get_DC(SR4_0S, ETS)
            # DC2_0 += get_DC(SR2_0S, TCS)
            # DC1_0 += get_DC(SR1_0S, WTS)

            # DC4_1 += get_DC(SR4_1S, ETS)
            # DC2_1 += get_DC(SR2_1S, TCS)
            # DC1_1 += get_DC(SR1_1S, WTS)


            # DC4_2 += get_DC(SR4_2S, ETS)
            # DC2_2 += get_DC(SR2_2S, TCS)
            # DC1_2 += get_DC(SR1_2S, WTS)


            # DC4_3 += get_DC(SR4_3S, ETS)
            # DC2_3 += get_DC(SR2_3S, TCS)
            # DC1_3 += get_DC(SR1_3S, WTS)

        
        
        se1,sp1,dc1,se2,sp2,dc2,se4,sp4,dc4 = SE1/p,SP1/p,DC1/p,SE2/p,SP2/p,DC2/p,SE4/p,SP4/p,DC4/p

        # dc_4_0,dc_2_0,dc_1_0=DC4_0/p,DC2_0/p,DC1_0/p
        # dc_4_1,dc_2_1,dc_1_1=DC4_1/p,DC2_1/p,DC1_1/p
        # dc_4_2,dc_2_2,dc_1_2=DC4_2/p,DC2_2/p,DC1_2/p
        # dc_4_3,dc_2_3,dc_1_3=DC4_3/p,DC2_3/p,DC1_3/p

        # print(dc1,dc2,dc4)

        # y = np.array(dc)
        # np.savetxt('/storage/homefs/zk22e821/MM/results/dc'+"_"+args.extra+"_"+args.fold+'.txt',y)


        # y = np.array(hd_et)
        # np.savetxt('/storage/homefs/zk22e821/MM/results/hd_et'+"_"+args.extra+"_"+args.fold+'.txt',y)
        # y = np.array(hd_tc)
        # np.savetxt('/storage/homefs/zk22e821/MM/results/hd_tc'+"_"+args.extra+"_"+args.fold+'.txt',y)
        # y = np.array(hd_wt)
        # np.savetxt('/storage/homefs/zk22e821/MM/results/hd_wt'+"_"+args.extra+"_"+args.fold+'.txt',y)

        # y = np.array([dc1,dc2,dc4,(dc1+dc2+dc4)/3])
        # print((dc1+dc2+dc4)/3)
        # y = np.array([dc_w,dc_t,dc_e])
        # np.savetxt('/storage/homefs/zk22e821/MM/augresults/dc'+"_"+args.extra+"_"+args.fold+"_"+args.aug+"_"+str(args.modal)+'.txt',y)

        HDW = np.array(hd_wt)
        y = HDW
        np.savetxt('/storage/homefs/zk22e821/MM/augresults/HDW'+"_"+args.extra+"_"+args.fold+"_"+args.aug+"_"+str(args.modal)+'.txt',y)
        HDT = np.array(hd_tc)
        y = HDT
        np.savetxt('/storage/homefs/zk22e821/MM/augresults/HDT'+"_"+args.extra+"_"+args.fold+"_"+args.aug+"_"+str(args.modal)+'.txt',y)
        HDE = np.array(hd_et)
        y = HDE
        np.savetxt('/storage/homefs/zk22e821/MM/augresults/HDE'+"_"+args.extra+"_"+args.fold+"_"+args.aug+"_"+str(args.modal)+'.txt',y)
           

