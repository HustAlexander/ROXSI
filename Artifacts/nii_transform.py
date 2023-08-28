import glob
import os
import h5py
import numpy as np
import SimpleITK as sitk
import random
from tqdm import tqdm
# from skimage import measure


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fold",type=str,default="0")
args = parser.parse_args()


slice_num = 0
cnt = 0

# data_root= './MICCAI_BraTS_2019_Data_Training/HGG'


data_root= '3Dtranformed_dataroot'+args.fold
label_root = "3Dlabelroot"+args.fold
# train_list = ['HGG__BraTS19_2013_10_1', 'HGG__BraTS19_2013_17_1', 'HGG__BraTS19_2013_25_1', 'HGG__BraTS19_CBICA_AAP_1', 'HGG__BraTS19_CBICA_ABY_1', 'HGG__BraTS19_CBICA_AMH_1', 'HGG__BraTS19_CBICA_ANI_1', 'HGG__BraTS19_CBICA_AOO_1', 'HGG__BraTS19_CBICA_AQQ_1', 'HGG__BraTS19_CBICA_AQR_1', 'HGG__BraTS19_CBICA_AQV_1', 'HGG__BraTS19_CBICA_AQY_1', 'HGG__BraTS19_CBICA_AQZ_1', 'HGG__BraTS19_CBICA_ASH_1', 'HGG__BraTS19_CBICA_ASN_1', 'HGG__BraTS19_CBICA_ASR_1', 'HGG__BraTS19_CBICA_ASU_1', 'HGG__BraTS19_CBICA_ASY_1', 'HGG__BraTS19_CBICA_ATB_1', 'HGG__BraTS19_CBICA_ATD_1', 'HGG__BraTS19_CBICA_ATF_1', 'HGG__BraTS19_CBICA_AUR_1', 'HGG__BraTS19_CBICA_AXJ_1', 'HGG__BraTS19_CBICA_AXM_1', 'HGG__BraTS19_CBICA_AXN_1', 'HGG__BraTS19_CBICA_AZD_1', 'HGG__BraTS19_CBICA_AZH_1', 'HGG__BraTS19_CBICA_BCF_1', 'HGG__BraTS19_CBICA_BFP_1', 'HGG__BraTS19_CBICA_BGO_1', 'HGG__BraTS19_CBICA_BHB_1', 'HGG__BraTS19_CBICA_BHV_1', 'HGG__BraTS19_CBICA_BHZ_1', 'HGG__BraTS19_CBICA_BKV_1', 'HGG__BraTS19_TCIA01_201_1', 'HGG__BraTS19_TCIA01_425_1', 'HGG__BraTS19_TCIA02_117_1', 'HGG__BraTS19_TCIA02_118_1', 'HGG__BraTS19_TCIA02_198_1', 'HGG__BraTS19_TCIA02_300_1', 'HGG__BraTS19_TCIA02_322_1', 'HGG__BraTS19_TCIA02_605_1', 'HGG__BraTS19_TCIA03_199_1', 'HGG__BraTS19_TCIA03_265_1', 'HGG__BraTS19_TCIA04_149_1', 'HGG__BraTS19_TCIA05_396_1', 'HGG__BraTS19_TCIA05_444_1', 'HGG__BraTS19_TCIA06_211_1', 'HGG__BraTS19_TCIA06_409_1', 'HGG__BraTS19_TCIA08_319_1', 'HGG__BraTS19_TCIA08_406_1', 'HGG__BraTS19_TMC_06290_1', 'HGG__BraTS19_TMC_06643_1', 'HGG__BraTS19_TMC_27374_1', 'LGG__BraTS19_2013_0_1', 'LGG__BraTS19_2013_16_1', 'LGG__BraTS19_2013_24_1', 'LGG__BraTS19_2013_8_1', 'LGG__BraTS19_TCIA09_493_1', 'LGG__BraTS19_TCIA10_202_1', 'LGG__BraTS19_TCIA10_261_1', 'LGG__BraTS19_TCIA10_307_1', 'LGG__BraTS19_TCIA10_387_1', 'LGG__BraTS19_TCIA12_101_1', 'LGG__BraTS19_TCIA13_634_1', 'LGG__BraTS19_TCIA13_650_1', 'LGG__BraTS19_TCIA13_653_1']
train_list = os.listdir(data_root)
train_list = sorted(train_list, key=lambda x: ((x.split('_')[-3])))
# train_list = sorted(train_list, key=lambda x: x)
i = 0

name = "000"
for case_ in train_list:
    # print(case_)
    
    # name = []
    # print(f'name：{case_}')
    # gg = case_.split("__")[0]
    # case = case_.split("__")[1]
    name_ = case_.split("000")[-2]
    if name_ == name:
        continue
    name = name_

    i+=1
    #print(i)
    
    # t1_path = os.path.join(data_root,gg, case,case+'_t1.nii.gz')
    # t1ce_path = os.path.join(data_root,gg,  case,case+'_t1ce.nii.gz')
    # t2_path = os.path.join(data_root,gg,  case,case+'_t2.nii.gz')
    # flair_path = os.path.join(data_root,gg, case,case+'_flair.nii.gz')

    t1_path = os.path.join(data_root,case_.split("000")[-2]+"0000.nii.gz")
    t1ce_path = os.path.join(data_root,case_.split("000")[-2]+"0001.nii.gz")
    t2_path = os.path.join(data_root,case_.split("000")[-2]+"0002.nii.gz")
    flair_path = os.path.join(data_root,case_.split("000")[-2]+"0003.nii.gz")

    msk_path = os.path.join(label_root,case_.split("_1_00")[-2]+"_1.nii.gz")

    t1_itk = sitk.ReadImage(t1_path)
    t1 = sitk.GetArrayFromImage(t1_itk)
    t1 = t1.astype(np.int16)
    
    t1ce_itk = sitk.ReadImage(t1ce_path)
    t1ce = sitk.GetArrayFromImage(t1ce_itk)
    t1ce = t1ce.astype(np.int16)

    t2_itk = sitk.ReadImage(t2_path)
    t2 = sitk.GetArrayFromImage(t2_itk)
    t2 = t2.astype(np.int16)

    flair_itk = sitk.ReadImage(flair_path)
    flair = sitk.GetArrayFromImage(flair_itk)
    flair = flair.astype(np.int16)


    msk_itk = sitk.ReadImage(msk_path)
    mask = sitk.GetArrayFromImage(msk_itk)



    for slice_ind in range(t1.shape[0]):
        f = h5py.File(
            '/storage/homefs/zk22e821/Dataset/Brats18_nn_transform/'+args.fold+'/{}_slice_{}.h5'.format(i, slice_ind), 'w')
        f.create_dataset(
            'flair', data=flair[slice_ind], compression='gzip')
        f.create_dataset(
            't1ce', data=t1ce[slice_ind], compression='gzip')
        f.create_dataset(
            't2', data=t2[slice_ind], compression='gzip')
        f.create_dataset(
            't1', data=t1[slice_ind], compression='gzip')
        # f.create_dataset(
        #     'Tmax', data=Tmax[slice_ind], compression='gzip')
        # f.create_dataset(
        #     'ADC', data=ADC[slice_ind], compression='gzip')
        f.create_dataset('label', data=mask[slice_ind], compression='gzip')
        # f.create_dataset(
        #     'lung_mask', data=lung_mask[slice_ind], compression='gzip')
        f.close()

        # name.append(f'{case_}_slice_{slice_ind}')
        slice_num += 1


print('Total {} slices'.format(slice_num))
