import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms as T
import nibabel as nib
import h5py
import random
import torchio.transforms as TT






class create20(data.Dataset):
    def __init__(self, data_root20,transforms=None, train=True, test=False,fold=0):
        self.test = test
        
        self.fold = int(fold)


        data0 = os.path.join(data_root20, "0")
        data1 = os.path.join(data_root20, "1")
        data2 = os.path.join(data_root20, "2")
        data3 = os.path.join(data_root20, "3")
        data4 = os.path.join(data_root20, "4")

        datas0 = [os.path.join(data0, png) for png in os.listdir(data0)]
        datas0 = sorted(datas0, key=lambda x: (int(x.split('_')[-3].split('/')[-1]),
                                     int(x.split('_')[-1].split('.')[-2])) )
        datas1 = [os.path.join(data1, png) for png in os.listdir(data1)]
        datas1 = sorted(datas1, key=lambda x: (int(x.split('_')[-3].split('/')[-1]),
                                     int(x.split('_')[-1].split('.')[-2])) )
        datas2 = [os.path.join(data2, png) for png in os.listdir(data2)]
        datas2 = sorted(datas2, key=lambda x: (int(x.split('_')[-3].split('/')[-1]),
                                     int(x.split('_')[-1].split('.')[-2])) )
        datas3 = [os.path.join(data3, png) for png in os.listdir(data3)]
        datas3 = sorted(datas3, key=lambda x: (int(x.split('_')[-3].split('/')[-1]),
                                     int(x.split('_')[-1].split('.')[-2])) )
        datas4 = [os.path.join(data4, png) for png in os.listdir(data4)]
        datas4 = sorted(datas4, key=lambda x: (int(x.split('_')[-3].split('/')[-1]),
                                     int(x.split('_')[-1].split('.')[-2])) )



        if self.fold == 0:
            data_train = datas1+datas2+datas3+datas4
            data_val = datas0
        elif self.fold == 1:
            data_train = datas0+datas2+datas3+datas4
            data_val = datas1
        elif self.fold == 2:
            data_train = datas0+datas1+datas3+datas4
            data_val = datas2
        elif self.fold == 3:
            data_train = datas0+datas1+datas2+datas4
            data_val = datas3
        elif self.fold == 4:
            data_train = datas0+datas1+datas2+datas3
            data_val = datas4

        
        self.train =train
        if self.train:
            self.datas = data_train


        else:
            self.datas = data_val


        self.transforms1 = T.CenterCrop(192) ####224 for Missformer#####
        self.transforms2 = T.RandomRotation(90)
        self.transforms3 = T.RandomHorizontalFlip(p=0.5)
        

    def __getitem__(self, index):

        

        input,label1,label2,label3,label4,Mask1,Mask2,Mask3,Mask4,Mask = self.read(index)


        return input, label1,label2,label4,Mask1,Mask2,Mask4,Mask,self.datas[index]
        

    def read(self,index):
        f = h5py.File(self.datas[index],'r')
        t1 =  np.expand_dims(f['t1'][:],0)
        t1ce =  np.expand_dims(f['t1ce'][:],0)
        t2 =  np.expand_dims(f['t2'][:],0)
        flair =  np.expand_dims(f['flair'][:],0)
        gt =  np.expand_dims(f['label'][:],0)

        t1 = self.normalize(t1)
        t1ce = self.normalize(t1ce)
        t2 = self.normalize(t2)
        flair = self.normalize(flair)

        input = np.concatenate((t1,t1ce,t2,flair),axis = 0)




        if self.train:
            data = np.concatenate((input,gt),axis = 0)
            data = torch.from_numpy(data).type(torch.FloatTensor)
            data = self.transforms1(data)
            data = self.transforms2(data)
            data = self.transforms3(data)
            

            input = data[:4,:,:]
            gt = data[4,:,:].unsqueeze(0)    
        else:
            input = torch.from_numpy(input).type(torch.FloatTensor)
            input = self.transforms1(input)
            gt = torch.from_numpy(gt*1.0).type(torch.FloatTensor)  
            gt = self.transforms1(gt)
        
        Mask1 = (gt==1).int()
        Mask2 = (gt==2).int()
        Mask3 = (gt==3).int()
        Mask4 = (gt==4).int()
        
        Mask = gt.int()
        Mask[gt == 4] = 3

        label1 = Mask1.sum()>0 
        label2 = Mask2.sum()>0 
        label3 = Mask3.sum()>0 
        label4 = Mask4.sum()>0 


        return input,label1,label2,label3,label4,Mask1,Mask2,Mask3,Mask4,Mask

    def normalize(self, data, smooth=1e-9):
        mean = data.mean()
        std = data.std()
        if (mean == 0) or (std == 0):
            return data

        data = (data - mean + smooth) / (std + smooth)
        return data


    def __len__(self):

        return len(self.datas)






class createval(data.Dataset):
    def __init__(self, transforms=None, train=True, test=False,fold=0, aug="biasfied", modal = 4):
        self.test = test
        
        self.fold = int(fold)
        self.modal = modal
    


        data = os.path.join("/storage/homefs/zk22e821/Dataset/Brats18_nn_transform/", str(self.fold))
        datas = [os.path.join(data, png) for png in os.listdir(data)]
        datas = sorted(datas, key=lambda x: (int(x.split('_')[-3].split('/')[-1]),
                                     int(x.split('_')[-1].split('.')[-2])) )



        
        self.train =train
        if self.train:
            self.datas = datas


        else:
            self.datas = datas


        self.transforms1 = T.CenterCrop(192)
        self.transforms2 = T.RandomRotation(90)
        self.transforms3 = T.RandomHorizontalFlip(p=0.5)
        RandomBiasField = TT.RandomBiasField(coefficients=0.5)
        noise = TT.RandomNoise(std=0.1)
        Ghost = TT.RandomGhosting(axes=1,intensity=1)
        Spike = TT.RandomSpike(intensity=3)
        motion = TT.RandomMotion(translation=500)
        if aug =="biasfield":
            self.augmentation = RandomBiasField
        elif aug == "spike":
            self.augmentation = Spike
        elif aug == "noise":
            self.augmentation = noise
        elif aug == "ghost":
            self.augmentation = Ghost
        elif aug == "motion":
            self.augmentation = motion
    

    def __getitem__(self, index):

        

        input,label1,label2,label3,label4,Mask1,Mask2,Mask3,Mask4,Mask = self.read(index)


        return input, label1,label2,label4,Mask1,Mask2,Mask4,Mask,self.datas[index]
        

    def read(self,index):
        f = h5py.File(self.datas[index],'r')
        t1 =  np.expand_dims(f['t1'][:],0)
        t1ce =  np.expand_dims(f['t1ce'][:],0)
        t2 =  np.expand_dims(f['t2'][:],0)
        flair =  np.expand_dims(f['flair'][:],0)
        gt =  np.expand_dims(f['label'][:],0)

        t1 = self.normalize(t1)
        t1ce = self.normalize(t1ce)
        t2 = self.normalize(t2)
        flair = self.normalize(flair)

        input = np.concatenate((t1,t1ce,t2,flair),axis = 0)




        if self.train:
            data = np.concatenate((input,gt),axis = 0)
            data = torch.from_numpy(data).type(torch.FloatTensor)
            data = self.transforms1(data)
            data = self.transforms2(data)
            data = self.transforms3(data)
            

            input = data[:4,:,:]
            gt = data[4,:,:].unsqueeze(0)    
        else:
            input = torch.from_numpy(input).type(torch.FloatTensor)

            gt = torch.from_numpy(gt*1.0).type(torch.FloatTensor)  
            gt = self.transforms1(gt)
            input = self.transforms1(input)
        
        Mask1 = (gt==1).int()
        Mask2 = (gt==2).int()
        Mask3 = (gt==3).int()
        Mask4 = (gt==4).int()

        Mask = gt.int()
        Mask[gt == 4] = 3

        label1 = Mask1.sum()>0 
        label2 = Mask2.sum()>0 
        label3 = Mask3.sum()>0 
        label4 = Mask4.sum()>0 


        return input,label1,label2,label3,label4,Mask1,Mask2,Mask3,Mask4,Mask

    def normalize(self, data, smooth=1e-9):
        mean = data.mean()
        std = data.std()
        if (mean == 0) or (std == 0):
            return data

        data = (data - mean + smooth) / (std + smooth)
        return data


    def __len__(self):

        return len(self.datas)







