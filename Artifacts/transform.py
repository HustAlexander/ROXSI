import SimpleITK as sitk
import os
import numpy as np
import torchio.transforms as TT
import torch
import random
from torchvision import transforms as T
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--aug",type=str)
parser.add_argument("--modal",type=int,default = 4)
parser.add_argument("--fold",type=str,default="0")
parser.add_argument("--nn",default=True)
args = parser.parse_args()


random.seed(309)
np.random.seed(309)
torch.manual_seed(309)
torch.cuda.manual_seed_all(309)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def normalize( data, smooth=1e-9):
    mean = data.mean()
    std = data.std()
    if (mean == 0) or (std == 0):
        return data

    data = (data - mean + smooth) / (std + smooth)
    return data

folder = "3D_dataroot"+args.fold
out_floder = "3Dtranformed_dataroot"+args.fold
train_list = os.listdir(folder)
train_list = sorted(train_list, key=lambda x: (x.split('_')[-3]))

RandomBiasField = TT.RandomBiasField(coefficients=1)
Ghost = TT.RandomGhosting(num_ghosts=(1,4),axes=1,intensity=(0.5,1))
Spike = TT.RandomSpike(num_spikes=(1,3),intensity=4)
motion = TT.RandomMotion(degrees=(10,40),num_transforms=6)
if args.aug =="biasfield":
    augmentation = RandomBiasField
elif args.aug == "spike":
    augmentation = Spike
elif args.aug == "motion":
    augmentation = motion
elif args.aug == "ghost":
    augmentation = Ghost
name = "000"
for case in train_list:
    name_ = case.split("000")[-2]
    if name_ == name:
        continue
    name = name_
    
    path0= os.path.join(folder,case.split("000")[-2]+"0000.nii.gz")
    outpath0 = os.path.join(out_floder,case.split("000")[-2]+"0000.nii.gz")
    path1= os.path.join(folder,case.split("000")[-2]+"0001.nii.gz")
    outpath1 = os.path.join(out_floder,case.split("000")[-2]+"0001.nii.gz")
    path2= os.path.join(folder,case.split("000")[-2]+"0002.nii.gz")
    outpath2 = os.path.join(out_floder,case.split("000")[-2]+"0002.nii.gz")
    path3= os.path.join(folder,case.split("000")[-2]+"0003.nii.gz")
    outpath3 = os.path.join(out_floder,case.split("000")[-2]+"0003.nii.gz")
    t0_itk = sitk.ReadImage(path0)
    t0 = sitk.GetArrayFromImage(t0_itk)
    t1_itk = sitk.ReadImage(path1)
    t1 = sitk.GetArrayFromImage(t1_itk)
    t2_itk = sitk.ReadImage(path2)
    t2 = sitk.GetArrayFromImage(t2_itk)
    t3_itk = sitk.ReadImage(path3)
    t3 = sitk.GetArrayFromImage(t3_itk)
    # t1 = t1.astype(np.int16)
    t0 = torch.from_numpy(t0).type(torch.FloatTensor).unsqueeze(0)
    t1 = torch.from_numpy(t1).type(torch.FloatTensor).unsqueeze(0)
    t2 = torch.from_numpy(t2).type(torch.FloatTensor).unsqueeze(0)
    t3 = torch.from_numpy(t3).type(torch.FloatTensor).unsqueeze(0)
    t0,t1,t2,t3 = normalize(t0),normalize(t1),normalize(t2),normalize(t3)

    

    if args.modal == 4:
        t0 = augmentation(t0)
        t1 = augmentation(t1)
        t2 = augmentation(t2)
        t3 = augmentation(t3)
        
        t = torch.cat((t0,t1,t2,t3),0)
        
    elif args.modal == 3:
        t = torch.cat((t0,t1,t2,t3),0)
        [a,b,c] = np.random.choice(4,3,replace =False)
        t[a,:,:,:] = augmentation(t[a,:,:,:].unsqueeze(0)).squeeze(0)
        t[b,:,:,:] = augmentation(t[b,:,:,:].unsqueeze(0)).squeeze(0)
        t[c,:,:,:] = augmentation(t[c,:,:,:].unsqueeze(0)).squeeze(0)
    elif args.modal == 2:
        t = torch.cat((t0,t1,t2,t3),0)
        [a,b,c] = np.random.choice(4,3,replace =False)
        t[a,:,:,:] = augmentation(t[a,:,:,:].unsqueeze(0)).squeeze(0)
        t[b,:,:,:] = augmentation(t[b,:,:,:].unsqueeze(0)).squeeze(0)
    elif args.modal == 1:
        t = torch.cat((t0,t1,t2,t3),0)
        [a,b,c] = np.random.choice(4,3,replace =False)
        t[a,:,:,:] = augmentation(t[a,:,:,:].unsqueeze(0)).squeeze(0)

    # if args.nn == True:
    #     t = t*(t>0)

    t = t*(t>0)
    t0 = t[0,:,:,:]
    t1 = t[1,:,:,:]
    t2 = t[2,:,:,:]
    t3 = t[3,:,:,:]
    
    t0 = t0 / (torch.max(t0)+1e-6)
    t1 = t1 / (torch.max(t1)+1e-6)
    t2 = t2 / (torch.max(t2)+1e-6)
    t3 = t3 / (torch.max(t3)+1e-6)
    

    t0 = (255*t0).numpy().astype(np.int16)
    t1 = (255*t1).numpy().astype(np.int16)
    t2 = (255*t2).numpy().astype(np.int16)
    t3 = (255*t3).numpy().astype(np.int16)
    # print(t1.mean(),t1.std())
    # t1 = t1.astype(np.int16)
    out0 = sitk.GetImageFromArray(t0)
    sitk.WriteImage(out0,outpath0)
    out1 = sitk.GetImageFromArray(t1)
    sitk.WriteImage(out1,outpath1)
    out2 = sitk.GetImageFromArray(t2)
    sitk.WriteImage(out2,outpath2)
    out3 = sitk.GetImageFromArray(t3)
    sitk.WriteImage(out3,outpath3)

    