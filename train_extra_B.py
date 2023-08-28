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
from model_seg import IU_Net_5_1,IU_Net_5_1_a,IU_Net_5,IU_Net_5_1_t,IU_Net_5_1_d,IU_Net_5_1_k,IU_Net_5_1_n
from Missformer.MISSFormer import MISSFormer
import logging
import sys
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F





random.seed(309)
np.random.seed(309)
torch.manual_seed(309)
torch.cuda.manual_seed_all(309)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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
    parser.add_argument("--fold",type=str,default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    outputs_dir = '/storage/homefs/zk22e821/MM'
    if not os.path.exists(outputs_dir):
        os.mkdir(outputs_dir)
    logs_dir = os.path.join(outputs_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)

    





    if args.extra == "5_1_a":
        model = IU_Net_5_1_a(feature_ch=int(args.in_channels),output_ch=3).cuda()
        logging.basicConfig(filename=logs_dir+'/'+args.training+"_"+args.extra+"_"+args.fold+'.txt', level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))

        writer = SummaryWriter(log_dir="./runs/"+args.training+args.extra+"_"+args.fold+'_extra')
    elif args.extra == "5":
        model = IU_Net_5(feature_ch=int(args.in_channels),output_ch=3).cuda()
        logging.basicConfig(filename=logs_dir+'/'+args.training+"_"+args.extra+"_"+args.fold+'.txt', level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))

        writer = SummaryWriter(log_dir="./runs/"+args.training+args.extra+"_"+args.fold+'_extra')

    elif args.extra == "5_1_d":
        model = IU_Net_5_1_d(feature_ch=int(args.in_channels),output_ch=3).cuda()
        logging.basicConfig(filename=logs_dir+'/'+args.training+"_"+args.extra+"_"+args.fold+'.txt', level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))

        writer = SummaryWriter(log_dir="./runs/"+args.training+args.extra+"_"+args.fold+'_extra')

    elif args.extra == "missformer":
        model = MISSFormer(num_classes=3).cuda()
        logging.basicConfig(filename=logs_dir+'/'+args.training+"_"+args.extra+"_"+args.fold+'.txt', level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))

        writer = SummaryWriter(log_dir="./runs/"+args.training+args.extra+"_"+args.fold+'_extra')




    criterion1 = DiceLoss()
    criterion2 = nn.L1Loss()

    train_data = dataset.create20(opt.data_root20, train=True,fold=args.fold)
    val_data = dataset.create20(opt.data_root20, train=False,fold=args.fold)

    val_dataloader = DataLoader(val_data, 1)

    lr=opt.lr

    optimizer=optim.Adam(model.parameters(),lr=opt.lr)
    # optimizer_D=optim.Adam(model.D.parameters(),lr=opt.lr)
    best_score = 0.7
    best_epoch = 0.0

    max_iter = 400


    from Training.seg_extra_B import seg_extra_B
    from Training.seg_m_B import seg_m_B
    from Training.seg_extra_align import seg_extra_align
    


    if args.training=="seg_extra_B":
        train = seg_extra_B
    elif args.training=="seg_m_B":
        train = seg_m_B



    for epoch in range(opt.max_epoch):


        model.train()
        l1,l2,l3 = 0,0,0

        length1 = 0
        length2 = 0
        length4 = 0
        SE1,SP1,DC1,SE2,SP2,DC2,SE4,SP4,DC4 = 0.,0.,0.,0.,0.,0.,0.,0.,0.
        train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True)
        
        for i, (data, label1,label2,label4,Mask1,Mask2,Mask4,Mask,data_name) in enumerate(train_dataloader):

            if i>max_iter:
                break

            input = Variable(data.cuda())
            Mask1 = Variable(Mask1.cuda()).float()
            Mask2 = Variable(Mask2.cuda()).float()
            Mask4 = Variable(Mask4.cuda()).float()

            WT = Mask1+Mask2+Mask4
            TC = Mask1+Mask4
            ET = Mask4

            # Mask = Variable(Mask.cuda()).long()

            GT = Variable(Mask.cuda()).squeeze(1).long()


            label1 = Variable(label1.cuda()).long()
            label2 = Variable(label2.cuda()).long()
            label4 = Variable(label4.cuda()).long()
            label = ((label1+label2+label4)>0).long()
        

            loss1,loss2,loss3,output = train(input,WT,TC,ET,model,optimizer)

            SR1 = output[:,0,:].unsqueeze(1)
            SR2 = output[:,1,:].unsqueeze(1)
            SR4 = output[:,2,:].unsqueeze(1)


            ####### evaluate on SLICE #########
            if label1.sum() > 0:
                SE1 += get_sensitivity(SR1, WT)
                SP1 += get_specificity(SR1, WT)
                DC1 += get_DC(SR1, WT)
                length1 +=1
                
            if label2.sum() > 0:
                SE2 += get_sensitivity(SR2, TC)
                SP2 += get_specificity(SR2, TC)
                DC2 += get_DC(SR2, TC)
                length2 +=1

            if label2.sum() > 0:
                SE4 += get_sensitivity(SR4, ET)
                SP4 += get_specificity(SR4, ET)
                DC4 += get_DC(SR4, ET)
                length4 +=1

            l1 +=loss1
            l2 +=loss2
            l3 +=loss3


        se1,sp1,dc1,se2,sp2,dc2,se4,sp4,dc4 = SE1/length1,SP1/length1,DC1/length1,SE2/length2,SP2/length2,DC2/length2,SE4/length4,SP4/length4,DC4/length4

        logging.info(
            'Epoch [%d/%d], Loss1: %.4f, Loss2: %.4f,Loss3: %.4f,lr: %.4f,\n[Training]\n SE1: %.4f, SP1: %.4f, DC1: %.4f\n SE2: %.4f, SP2: %.4f, DC2: %.4f\n  SE4: %.4f, SP4: %.4f, DC4: %.4f\n' % (
                epoch + 1, opt.max_epoch,l1,l2,l3,10*lr,
                se1,sp1,dc1,
                se2,sp2,dc2,
                se4,sp4,dc4))

        writer.add_scalar(tag="loss/train", scalar_value=l1,
                        global_step=epoch )

        writer.add_scalar(tag="dc4/train", scalar_value=dc4,
                            global_step=epoch )

        writer.add_scalar(tag="dc2/train", scalar_value=dc2,
                            global_step=epoch )
        
        writer.add_scalar(tag="dc1/train", scalar_value=dc1,
                            global_step=epoch )

        writer.add_scalar(tag="dc/train", scalar_value=(dc1+dc2+dc4)/3,
                            global_step=epoch )

        if epoch + 1 == 10 or epoch + 1 == 20 or epoch + 1 == 30 or epoch + 1 == 40 or epoch + 1 == 60 or epoch + 1 == 80 or epoch + 1 == 100\
            or epoch + 1 == 120 or epoch + 1 == 140: 
            lr = lr*0.8
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        


        model.eval()
        l = 0
        total=0
        SE1,SP1,DC1,SE2,SP2,DC2,SE4,SP4,DC4 = 0.,0.,0.,0.,0.,0.,0.,0.,0.
        SE4_1,SP4_1,DC4_1,SE4_0,SP4_0,DC4_0,SE4_2,SP4_2,DC4_2,SE4_3,SP4_3,DC4_3 = 0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.
        DC4_0,DC2_0,DC1_0,DC4_1,DC2_1,DC1_1,DC4_2,DC2_2,DC1_2,DC4_3,DC2_3,DC1_3 = 0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.

        SE_WT,SP_WT,DC_WT,SE_TC,SP_TC,DC_TC = 0.,0.,0.,0.,0.,0.

        id = 0
        s = 1
        p = 0

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

                if args.training =="seg_m_B":
                    out = model(input)
                    out0,out1,out2,out3 = 0*out,0*out,0*out,0*out
                else:
                    out,out0,out1,out2,out3 = model(input)


                
                SR1 = torch.sigmoid(out[:,0,:])
                SR2 = torch.sigmoid(out[:,1,:])
                SR4 = torch.sigmoid(out[:,2,:])

                SR1 = SR1.unsqueeze(1)
                SR2 = SR2.unsqueeze(1)
                SR4 = SR4.unsqueeze(1)


                out0_1 = torch.sigmoid(out0[:,0,:]).unsqueeze(1)
                out0_2 = torch.sigmoid(out0[:,1,:]).unsqueeze(1)
                out0_4 = torch.sigmoid(out0[:,2,:]).unsqueeze(1)


                out1_1 = torch.sigmoid(out1[:,0,:]).unsqueeze(1)
                out1_2 = torch.sigmoid(out1[:,1,:]).unsqueeze(1)
                out1_4 = torch.sigmoid(out1[:,2,:]).unsqueeze(1)

                out2_1 = torch.sigmoid(out2[:,0,:]).unsqueeze(1)
                out2_2 = torch.sigmoid(out2[:,1,:]).unsqueeze(1)
                out2_4 = torch.sigmoid(out2[:,2,:]).unsqueeze(1)



                out3_1 = torch.sigmoid(out3[:,0,:]).unsqueeze(1)
                out3_2 = torch.sigmoid(out3[:,1,:]).unsqueeze(1)
                out3_4 = torch.sigmoid(out3[:,2,:]).unsqueeze(1)



                loss1 = criterion1(SR1,WT)+criterion1(SR2,TC)+criterion1(SR4,ET)
                l += loss1

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

                            DC4_0 += get_DC(SR4_0S, ETS)
                            DC2_0 += get_DC(SR2_0S, TCS)
                            DC1_0 += get_DC(SR1_0S, WTS)

                            DC4_1 += get_DC(SR4_1S, ETS)
                            DC2_1 += get_DC(SR2_1S, TCS)
                            DC1_1 += get_DC(SR1_1S, WTS)


                            DC4_2 += get_DC(SR4_2S, ETS)
                            DC2_2 += get_DC(SR2_2S, TCS)
                            DC1_2 += get_DC(SR1_2S, WTS)


                            DC4_3 += get_DC(SR4_3S, ETS)
                            DC2_3 += get_DC(SR2_3S, TCS)
                            DC1_3 += get_DC(SR1_3S, WTS)
                            


                    id = data_name[0].split('_')[-3].split('/')[-1]
                    SR1S,WTS,SR2S,TCS,SR4S,ETS = SR1,WT,SR2,TC,SR4,ET
                    SR4_0S,SR4_1S,SR4_2S,SR4_3S = out0_4,out1_4,out2_4,out3_4
                    SR2_0S,SR2_1S,SR2_2S,SR2_3S = out0_2,out1_2,out2_2,out3_2
                    SR1_0S,SR1_1S,SR1_2S,SR1_3S = out0_1,out1_1,out2_1,out3_1

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

                    SR4_0S = torch.cat((SR4_0S,out0_4),1)
                    SR4_1S = torch.cat((SR4_1S,out1_4),1)
                    SR4_2S = torch.cat((SR4_2S,out2_4),1)
                    SR4_3S = torch.cat((SR4_3S,out3_4),1)
                    SR2_0S = torch.cat((SR2_0S,out0_2),1)
                    SR2_1S = torch.cat((SR2_1S,out1_2),1)
                    SR2_2S = torch.cat((SR2_2S,out2_2),1)
                    SR2_3S = torch.cat((SR2_3S,out3_2),1)
                    SR1_0S = torch.cat((SR1_0S,out0_1),1)
                    SR1_1S = torch.cat((SR1_1S,out1_1),1)
                    SR1_2S = torch.cat((SR1_2S,out2_1),1)
                    SR1_3S = torch.cat((SR1_3S,out3_1),1)
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


                DC4_0 += get_DC(SR4_0S, ETS)
                DC2_0 += get_DC(SR2_0S, TCS)
                DC1_0 += get_DC(SR1_0S, WTS)

                DC4_1 += get_DC(SR4_1S, ETS)
                DC2_1 += get_DC(SR2_1S, TCS)
                DC1_1 += get_DC(SR1_1S, WTS)


                DC4_2 += get_DC(SR4_2S, ETS)
                DC2_2 += get_DC(SR2_2S, TCS)
                DC1_2 += get_DC(SR1_2S, WTS)


                DC4_3 += get_DC(SR4_3S, ETS)
                DC2_3 += get_DC(SR2_3S, TCS)
                DC1_3 += get_DC(SR1_3S, WTS)

            
            
            se1,sp1,dc1,se2,sp2,dc2,se4,sp4,dc4 = SE1/p,SP1/p,DC1/p,SE2/p,SP2/p,DC2/p,SE4/p,SP4/p,DC4/p

            dc_4_0,dc_2_0,dc_1_0=DC4_0/p,DC2_0/p,DC1_0/p
            dc_4_1,dc_2_1,dc_1_1=DC4_1/p,DC2_1/p,DC1_1/p
            dc_4_2,dc_2_2,dc_1_2=DC4_2/p,DC2_2/p,DC1_2/p
            dc_4_3,dc_2_3,dc_1_3=DC4_3/p,DC2_3/p,DC1_3/p

            score = (dc1+dc2+dc4)/3
            
            writer.add_scalar(tag="loss/val", scalar_value=l,
                            global_step=epoch)

            writer.add_scalar(tag="dc4/val", scalar_value=dc4,
                            global_step=epoch)

            writer.add_scalar(tag="dc2/val", scalar_value=dc2,
                            global_step=epoch)
            
            writer.add_scalar(tag="dc1/val", scalar_value=dc1,
                            global_step=epoch)
            
            writer.add_scalar(tag="dc/val", scalar_value=(dc1+dc2+dc4)/3,
                            global_step=epoch)
            
            logging.info(
                '[val] Loss: %.4f \n SE1: %.4f, SP1: %.4f, DC1: %.4f\n SE2: %.4f, SP2: %.4f, DC2: %.4f\n  SE4: %.4f, SP4: %.4f, DC4: %.4f\n' % (
                    l,
                    se1,sp1,dc1,
                    se2,sp2,dc2,
                    se4,sp4,dc4))
            logging.info(
                '[evaluation_ex0] DC_4_0: %.4f, DC_2_0: %.4f, DC_1_0: %.4f\n ' % (
                    dc_4_0,dc_2_0,dc_1_0))
            logging.info(
                '[evaluation_ex1] DC_4_1: %.4f, DC_2_1: %.4f, DC_1_1: %.4f\n ' % (
                    dc_4_1,dc_2_1,dc_1_1))
            logging.info(
                '[evaluation_ex2] DC_4_2: %.4f, DC_2_2: %.4f, DC_1_2: %.4f\n ' % (
                    dc_4_2,dc_2_2,dc_1_2))
            logging.info(
                '[evaluation_ex3] DC_4_3: %.4f, DC_2_3: %.4f, DC_1_3: %.4f\n ' % (
                    dc_4_3,dc_2_3,dc_1_3))



            # Save Best model
            if score > best_score:
                best_score = score
                best_epoch = epoch+1
                logging.info('Best model score : %.4f \n' % (best_score))
                best_Net = model.state_dict()
                model_dir = '/storage/homefs/zk22e821/MM/models/continue'
                if not os.path.exists(model_dir):
                    os.mkdir(model_dir)
                torch.save(best_Net, model_dir+'/'+args.training+"_"+args.extra+"_"+args.fold+'.ckpt')

            logging.info('Best model epoch & score : %.1f,%.4f \n' % (best_epoch,best_score))


    writer.close()
