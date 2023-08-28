import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch
from vicreg import VICReg,VICReg_w,VICReg_a


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True), 
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
            nn.ConvTranspose2d(ch_in,ch_in,kernel_size=3,stride=2,padding=1, output_padding=1),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x




class IEncoder_5(nn.Module):
    def __init__(self, img_ch, feature_ch) -> None:
        super(IEncoder_5, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 =  conv_block(img_ch, feature_ch)
        self.Conv2 =  conv_block(feature_ch, feature_ch*2)
        self.Conv3 =  conv_block(feature_ch*2, feature_ch*4)
        self.Conv4 =  conv_block(feature_ch*4, feature_ch*8)
        self.Conv5 =  conv_block(feature_ch*8, feature_ch*16)


    def forward(self, x):
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)   

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)  

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)  


        return x1,x2,x3,x4,x5


class IEncoder5(nn.Module):
    def __init__(self, output_ch, feature_ch) -> None:
        super(IEncoder5, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv5 =  conv_block(feature_ch*8, feature_ch*16)

    def forward(self, x4):
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)  

        return x5



class IDecoder(nn.Module):
    def __init__(self, output_ch, feature_ch) -> None:
        super(IDecoder, self).__init__()



        self.Up5 = up_conv(ch_in=feature_ch*4, ch_out=feature_ch*2)
        self.Up_conv5 = conv_block(ch_in=feature_ch*4, ch_out=feature_ch*2)

        self.Up4 = up_conv(ch_in=feature_ch*2, ch_out=feature_ch*1)
        self.Up_conv4 = conv_block(ch_in=feature_ch*2, ch_out=feature_ch*1)

        self.Up3 = up_conv(ch_in=feature_ch*1, ch_out=int(feature_ch*0.5))
        self.Up_conv3 = conv_block(ch_in=feature_ch*1, ch_out=int(feature_ch*0.5))


        self.Up2 = up_conv(ch_in=int(feature_ch*0.5), ch_out=int(feature_ch*0.25))
        self.Up_conv2 = conv_block(ch_in=int(feature_ch*0.5), ch_out=int(feature_ch*0.25))

        
        self.Conv_1x1_d = nn.Conv2d(int(feature_ch*0.25), output_ch, kernel_size=1, stride=1, padding=0)



    def forward(self, x1,x2,x3,x4,x5):
        d15 = self.Up5(x5)
        # print(d15.size(),x4.size())
        d15 = torch.cat((x4, d15), dim=1)
        d15 = self.Up_conv5(d15)

        d14 = self.Up4(d15)
        d14 = torch.cat((x3, d14), dim=1)
        d14 = self.Up_conv4(d14)

        # x3 = self.conv_d13(x3)
        d13 = self.Up3(d14)
        d13 = torch.cat((x2, d13), dim=1)
        d13 = self.Up_conv3(d13)

        d12 = self.Up2(d13)
        d12 = torch.cat((x1, d12), dim=1)
        d12 = self.Up_conv2(d12)
        

        d11 = self.Conv_1x1_d(d12)


        return d11





class co_att_q(nn.Module):
    def __init__(self, feature_ch) -> None:
        super(co_att_q, self).__init__()

        self.conv_e = nn.Sequential(
            nn.Conv2d(feature_ch,feature_ch*4, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(feature_ch*4),
            nn.ReLU()
        )
        self.conv_s = nn.Sequential(
            nn.Conv2d(feature_ch*4,feature_ch, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(feature_ch),
            nn.ReLU()
        )

        self.conv_q1 = nn.Conv2d(feature_ch,feature_ch,kernel_size=1,stride=1,padding=0,bias=False)
        self.conv_q2 = nn.Conv2d(feature_ch,feature_ch,kernel_size=1,stride=1,padding=0,bias=False)

        self.conv_v1 = conv_block(feature_ch,feature_ch)

        self.conv_v2 = conv_block(feature_ch,feature_ch)  


    def forward(self, x1,x2):

        x2_o = x2
        x2 = self.conv_s(x2) 
        x1_q = self.conv_q1(x1)
        x2_q = self.conv_q2(x2)
        x1_v = self.conv_v1(x1)
        x2_v = self.conv_v2(x2)

        x1_q_flat=x1_q.view(-1, x1.size()[1], x1.size()[2]*x1.size()[3])
        x2_q_flat=x2_q.view(-1, x2.size()[1], x2.size()[2]*x2.size()[3])

        x1_v_flat=x1_v.view(-1, x1.size()[1], x1.size()[2]*x1.size()[3])
        x2_v_flat=x2_v.view(-1, x2.size()[1], x2.size()[2]*x2.size()[3])

        x1_q_t = torch.transpose(x1_q_flat,1,2).contiguous()

        A = torch.bmm(x1_q_t, x2_q_flat)
        A = F.softmax(A, dim = 1)
        B = F.softmax(torch.transpose(A,1,2),dim=1)

        # x2_att = torch.bmm(x1_v_flat, A).contiguous()
        x1_att = torch.bmm(x2_v_flat, B).contiguous()
        x1_att = x1_att.view(-1, x2.size()[1], x2.size()[2], x2.size()[3])
        # x2_att = x2_att.view(-1, x2.size()[1], x2.size()[2], x2.size()[3])

        # x2_att = self.conv_e(x2_att)

        input1_att = x1_att+x1  
        input2_att = x2_o
    
        return input1_att,input2_att


class IU_Net_5(nn.Module):
    def __init__(self, img_ch=1, feature_ch=64, output_ch=1):
        super(IU_Net_5, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.E0 =  IEncoder_5(img_ch, int(feature_ch/4))
        self.E1 =  IEncoder_5(img_ch, int(feature_ch/4))
        self.E2 =  IEncoder_5(img_ch, int(feature_ch/4))
        self.E3 =  IEncoder_5(img_ch, int(feature_ch/4))
    


        self.D = IDecoder(output_ch, feature_ch*4)

        
    def forward(self, x):

        x01,x02,x03,x04,x05 = self.E0(x[:,0,:].unsqueeze(1))
        x11,x12,x13,x14,x15 = self.E1(x[:,1,:].unsqueeze(1))
        x21,x22,x23,x24,x25 = self.E2(x[:,2,:].unsqueeze(1))
        x31,x32,x33,x34,x35 = self.E3(x[:,3,:].unsqueeze(1))


        x1 = torch.cat((x01,x11,x21,x31),dim=1)
        x2 = torch.cat((x02,x12,x22,x32),dim=1)
        x3 = torch.cat((x03,x13,x23,x33),dim=1)
        x4 = torch.cat((x04,x14,x24,x34),dim=1)
        x5 = torch.cat((x05,x15,x25,x35),dim=1)

        d1 = self.D(x1,x2,x3,x4,x5)


        return d1



class IU_Net_5_1_a(nn.Module):
    def __init__(self, img_ch=1, feature_ch=64, output_ch=1):
        super(IU_Net_5_1_a, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.E0 =  IEncoder_5(img_ch, int(feature_ch/4))
        self.E1 =  IEncoder_5(img_ch, int(feature_ch/4))
        self.E2 =  IEncoder_5(img_ch, int(feature_ch/4))
        self.E3 =  IEncoder_5(img_ch, int(feature_ch/4))
        self.E4 =  IEncoder_5(img_ch, int(feature_ch/4))
        # self.E5 = IEncoder5(img_ch, int(feature_ch/4))
    
        # self.D5 = IDecoder5(output_ch, feature_ch)
        self.D0 = IDecoder(output_ch, feature_ch)
        self.D1 = IDecoder(output_ch, feature_ch)
        self.D2 = IDecoder(output_ch, feature_ch)
        self.D3 = IDecoder(output_ch, feature_ch)

        self.D = IDecoder(output_ch, feature_ch*4)

        # self.conv_q_4 = nn.Conv2d(feature_ch*2,feature_ch*2,kernel_size=1,stride=1,padding=0,bias=True)

        self.co_att05 = co_att_q(feature_ch*4)
        self.co_att15 = co_att_q(feature_ch*4)
        self.co_att25 = co_att_q(feature_ch*4)
        self.co_att35 = co_att_q(feature_ch*4)

        # self.co_att04 = co_att_q(feature_ch*2)
        # self.co_att14 = co_att_q(feature_ch*2)
        # self.co_att24 = co_att_q(feature_ch*2)
        # self.co_att34 = co_att_q(feature_ch*2)




    def forward(self, x):

        x01,x02,x03,x04,x05 = self.E0(x[:,0,:].unsqueeze(1))
        x11,x12,x13,x14,x15 = self.E1(x[:,1,:].unsqueeze(1))
        x21,x22,x23,x24,x25 = self.E2(x[:,2,:].unsqueeze(1))
        x31,x32,x33,x34,x35 = self.E3(x[:,3,:].unsqueeze(1))


        x1 = torch.cat((x01,x11,x21,x31),dim=1)
        x2 = torch.cat((x02,x12,x22,x32),dim=1)
        x3 = torch.cat((x03,x13,x23,x33),dim=1)
        x4 = torch.cat((x04,x14,x24,x34),dim=1)
        x5 = torch.cat((x05,x15,x25,x35),dim=1)


        x05,_= self.co_att05(x05,x5)
        x15,_= self.co_att15(x15,x5)
        x25,_= self.co_att25(x25,x5)
        x35,_= self.co_att35(x35,x5)

        # x04,_= self.co_att04(x04,x4)
        # x14,_= self.co_att14(x14,x4)
        # x24,_= self.co_att24(x24,x4)
        # x34,_= self.co_att34(x34,x4)

        x5 = torch.cat((x05,x15,x25,x35),dim=1)
        # x4 = torch.cat((x04,x14,x24,x34),dim=1)



        d01 = self.D0(x01,x02,x03,x04,x05)
        d11 = self.D1(x11,x12,x13,x14,x15)
        d21 = self.D2(x21,x22,x23,x24,x25)
        d31 = self.D3(x31,x32,x33,x34,x35)


        d1 = self.D(x1,x2,x3,x4,x5)

        return d1,d01,d11,d21,d31

class IU_Net_5_1_a_val(nn.Module):
    def __init__(self, img_ch=1, feature_ch=64, output_ch=1):
        super(IU_Net_5_1_a_val, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.E0 =  IEncoder_5(img_ch, int(feature_ch/4))
        self.E1 =  IEncoder_5(img_ch, int(feature_ch/4))
        self.E2 =  IEncoder_5(img_ch, int(feature_ch/4))
        self.E3 =  IEncoder_5(img_ch, int(feature_ch/4))
        self.E4 =  IEncoder_5(img_ch, int(feature_ch/4))
        # self.E5 = IEncoder5(img_ch, int(feature_ch/4))
    

        self.D = IDecoder(output_ch, feature_ch*4)

        # self.conv_q_4 = nn.Conv2d(feature_ch*2,feature_ch*2,kernel_size=1,stride=1,padding=0,bias=True)

        self.co_att05 = co_att_q(feature_ch*4)
        self.co_att15 = co_att_q(feature_ch*4)
        self.co_att25 = co_att_q(feature_ch*4)
        self.co_att35 = co_att_q(feature_ch*4)




    def forward(self, x):

        x01,x02,x03,x04,x05 = self.E0(x[:,0,:].unsqueeze(1))
        x11,x12,x13,x14,x15 = self.E1(x[:,1,:].unsqueeze(1))
        x21,x22,x23,x24,x25 = self.E2(x[:,2,:].unsqueeze(1))
        x31,x32,x33,x34,x35 = self.E3(x[:,3,:].unsqueeze(1))


        x1 = torch.cat((x01,x11,x21,x31),dim=1)
        x2 = torch.cat((x02,x12,x22,x32),dim=1)
        x3 = torch.cat((x03,x13,x23,x33),dim=1)
        x4 = torch.cat((x04,x14,x24,x34),dim=1)
        x5 = torch.cat((x05,x15,x25,x35),dim=1)


        x05,_= self.co_att05(x05,x5)
        x15,_= self.co_att15(x15,x5)
        x25,_= self.co_att25(x25,x5)
        x35,_= self.co_att35(x35,x5)


        x5 = torch.cat((x05,x15,x25,x35),dim=1)
        # x4 = torch.cat((x04,x14,x24,x34),dim=1)





        d1 = self.D(x1,x2,x3,x4,x5)

        return d1

class IU_Net_5_1_d(nn.Module):
    def __init__(self, img_ch=1, feature_ch=64, output_ch=1):
        super(IU_Net_5_1_d, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.E0 =  IEncoder_5(img_ch, int(feature_ch/4))
        self.E1 =  IEncoder_5(img_ch, int(feature_ch/4))
        self.E2 =  IEncoder_5(img_ch, int(feature_ch/4))
        self.E3 =  IEncoder_5(img_ch, int(feature_ch/4))
        self.E4 =  IEncoder_5(img_ch, int(feature_ch/4))

        self.D0 = IDecoder(output_ch, feature_ch)
        self.D1 = IDecoder(output_ch, feature_ch)
        self.D2 = IDecoder(output_ch, feature_ch)
        self.D3 = IDecoder(output_ch, feature_ch)

        self.D = IDecoder(output_ch, feature_ch*4)


        self.co_att05 = co_att_q(feature_ch*4)
        self.co_att15 = co_att_q(feature_ch*4)
        self.co_att25 = co_att_q(feature_ch*4)
        self.co_att35 = co_att_q(feature_ch*4)

        self.co_att04 = co_att_q(feature_ch*2)
        self.co_att14 = co_att_q(feature_ch*2)
        self.co_att24 = co_att_q(feature_ch*2)
        self.co_att34 = co_att_q(feature_ch*2)

        self.vic5 = VICReg_w(feature_ch*4,3,1,3)
        # self.vic4 = VICReg_w(feature_ch*2,3,0.1,3)
        # self.vic3 = VICReg_w(feature_ch,3,1,3)


    def forward(self, x):

        x01,x02,x03,x04,x05 = self.E0(x[:,0,:].unsqueeze(1))
        x11,x12,x13,x14,x15 = self.E1(x[:,1,:].unsqueeze(1))
        x21,x22,x23,x24,x25 = self.E2(x[:,2,:].unsqueeze(1))
        x31,x32,x33,x34,x35 = self.E3(x[:,3,:].unsqueeze(1))


        x1 = torch.cat((x01,x11,x21,x31),dim=1)
        x2 = torch.cat((x02,x12,x22,x32),dim=1)
        x3 = torch.cat((x03,x13,x23,x33),dim=1)
        x4 = torch.cat((x04,x14,x24,x34),dim=1)
        x5 = torch.cat((x05,x15,x25,x35),dim=1)

        

        x05,_= self.co_att05(x05,x5)
        x15,_= self.co_att15(x15,x5)
        x25,_= self.co_att25(x25,x5)
        x35,_= self.co_att35(x35,x5)

        # x04,_= self.co_att04(x04,x4)
        # x14,_= self.co_att14(x14,x4)
        # x24,_= self.co_att24(x24,x4)
        # x34,_= self.co_att34(x34,x4)

        x5 = torch.cat((x05,x15,x25,x35),dim=1)
        # x4 = torch.cat((x04,x14,x24,x34),dim=1)



        d01 = self.D0(x01,x02,x03,x04,x05)
        d11 = self.D1(x11,x12,x13,x14,x15)
        d21 = self.D2(x21,x22,x23,x24,x25)
        d31 = self.D3(x31,x32,x33,x34,x35)


        d1 = self.D(x1,x2,x3,x4,x5)
        
        loss_vic = self.vic5(x05,x15,x25,x35)
        # loss_vic += self.vic4(x04,x14,x24,x34)
        # loss_vic += self.vic3(x03,x13,x23,x33)

        return d1,d01,d11,d21,d31,loss_vic
        # return d1


class IU_Net_5_1_d_val(nn.Module):
    def __init__(self, img_ch=1, feature_ch=64, output_ch=1):
        super(IU_Net_5_1_d_val, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.E0 =  IEncoder_5(img_ch, int(feature_ch/4))
        self.E1 =  IEncoder_5(img_ch, int(feature_ch/4))
        self.E2 =  IEncoder_5(img_ch, int(feature_ch/4))
        self.E3 =  IEncoder_5(img_ch, int(feature_ch/4))
        self.E4 =  IEncoder_5(img_ch, int(feature_ch/4))


        self.D = IDecoder(output_ch, feature_ch*4)


        self.co_att05 = co_att_q(feature_ch*4)
        self.co_att15 = co_att_q(feature_ch*4)
        self.co_att25 = co_att_q(feature_ch*4)
        self.co_att35 = co_att_q(feature_ch*4)

        self.co_att04 = co_att_q(feature_ch*2)
        self.co_att14 = co_att_q(feature_ch*2)
        self.co_att24 = co_att_q(feature_ch*2)
        self.co_att34 = co_att_q(feature_ch*2)

        # self.vic5 = VICReg_w(feature_ch*4,3,1,3)
        # self.vic4 = VICReg_w(feature_ch*2,3,0.1,3)
        # self.vic3 = VICReg_w(feature_ch,3,1,3)


    def forward(self, x):

        x01,x02,x03,x04,x05 = self.E0(x[:,0,:].unsqueeze(1))
        x11,x12,x13,x14,x15 = self.E1(x[:,1,:].unsqueeze(1))
        x21,x22,x23,x24,x25 = self.E2(x[:,2,:].unsqueeze(1))
        x31,x32,x33,x34,x35 = self.E3(x[:,3,:].unsqueeze(1))


        x1 = torch.cat((x01,x11,x21,x31),dim=1)
        x2 = torch.cat((x02,x12,x22,x32),dim=1)
        x3 = torch.cat((x03,x13,x23,x33),dim=1)
        x4 = torch.cat((x04,x14,x24,x34),dim=1)
        x5 = torch.cat((x05,x15,x25,x35),dim=1)

        

        x05,_= self.co_att05(x05,x5)
        x15,_= self.co_att15(x15,x5)
        x25,_= self.co_att25(x25,x5)
        x35,_= self.co_att35(x35,x5)

        # x04,_= self.co_att04(x04,x4)
        # x14,_= self.co_att14(x14,x4)
        # x24,_= self.co_att24(x24,x4)
        # x34,_= self.co_att34(x34,x4)

        x5 = torch.cat((x05,x15,x25,x35),dim=1)
        # x4 = torch.cat((x04,x14,x24,x34),dim=1)



        # d01 = self.D0(x01,x02,x03,x04,x05)
        # d11 = self.D1(x11,x12,x13,x14,x15)
        # d21 = self.D2(x21,x22,x23,x24,x25)
        # d31 = self.D3(x31,x32,x33,x34,x35)


        d1 = self.D(x1,x2,x3,x4,x5)
        
        # loss_vic = self.vic5(x05,x15,x25,x35)
        # loss_vic += self.vic4(x04,x14,x24,x34)
        # loss_vic += self.vic3(x03,x13,x23,x33)

        return d1
        # return d1

