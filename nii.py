import glob
import os
import h5py
import numpy as np
import SimpleITK as sitk
import random
from tqdm import tqdm
# from skimage import measure




slice_num = 0
cnt = 0

# data_root= './MICCAI_BraTS_2019_Data_Training/HGG'


data_root= '/storage/homefs/zk22e821/Dataset/MICCAI_BraTS_2019_Data_Training/'


####train_list divided by nnUNet######
train_list = ['HGG__BraTS19_2013_13_1', 'HGG__BraTS19_2013_19_1', 'HGG__BraTS19_2013_27_1', 'HGG__BraTS19_CBICA_AAG_1', 'HGG__BraTS19_CBICA_ALN_1', 'HGG__BraTS19_CBICA_ANV_1', 'HGG__BraTS19_CBICA_AOH_1', 'HGG__BraTS19_CBICA_APK_1', 'HGG__BraTS19_CBICA_APR_1', 'HGG__BraTS19_CBICA_AQG_1', 'HGG__BraTS19_CBICA_AQP_1', 'HGG__BraTS19_CBICA_ARZ_1', 'HGG__BraTS19_CBICA_ASF_1', 'HGG__BraTS19_CBICA_ASG_1', 'HGG__BraTS19_CBICA_ATP_1', 'HGG__BraTS19_CBICA_ATX_1', 'HGG__BraTS19_CBICA_AUA_1', 'HGG__BraTS19_CBICA_AVJ_1', 'HGG__BraTS19_CBICA_AVV_1', 'HGG__BraTS19_CBICA_AWG_1', 'HGG__BraTS19_CBICA_AXL_1', 'HGG__BraTS19_CBICA_AXQ_1', 'HGG__BraTS19_CBICA_BAN_1', 'HGG__BraTS19_CBICA_BBG_1', 'HGG__BraTS19_CBICA_BGE_1', 'HGG__BraTS19_CBICA_BHQ_1', 'HGG__BraTS19_CBICA_BIC_1', 'HGG__BraTS19_CBICA_BNR_1', 'HGG__BraTS19_TCIA01_131_1', 'HGG__BraTS19_TCIA01_147_1', 'HGG__BraTS19_TCIA01_180_1', 'HGG__BraTS19_TCIA01_190_1', 'HGG__BraTS19_TCIA01_221_1', 'HGG__BraTS19_TCIA01_335_1', 'HGG__BraTS19_TCIA01_411_1', 'HGG__BraTS19_TCIA02_151_1', 'HGG__BraTS19_TCIA02_321_1', 'HGG__BraTS19_TCIA02_331_1', 'HGG__BraTS19_TCIA02_368_1', 'HGG__BraTS19_TCIA02_471_1', 'HGG__BraTS19_TCIA03_257_1', 'HGG__BraTS19_TCIA03_474_1', 'HGG__BraTS19_TCIA04_111_1', 'HGG__BraTS19_TCIA04_328_1', 'HGG__BraTS19_TCIA04_343_1', 'HGG__BraTS19_TCIA05_277_1', 'HGG__BraTS19_TCIA05_478_1', 'HGG__BraTS19_TCIA06_165_1', 'HGG__BraTS19_TCIA08_105_1', 'HGG__BraTS19_TCIA08_280_1', 'HGG__BraTS19_TMC_15477_1', 'HGG__BraTS19_TMC_21360_1', 'HGG__BraTS19_TMC_30014_1', 'LGG__BraTS19_TCIA09_428_1', 'LGG__BraTS19_TCIA10_175_1', 'LGG__BraTS19_TCIA10_276_1', 'LGG__BraTS19_TCIA10_393_1', 'LGG__BraTS19_TCIA10_408_1', 'LGG__BraTS19_TCIA10_410_1', 'LGG__BraTS19_TCIA10_449_1', 'LGG__BraTS19_TCIA10_490_1', 'LGG__BraTS19_TCIA10_625_1', 'LGG__BraTS19_TCIA10_637_1', 'LGG__BraTS19_TCIA12_249_1', 'LGG__BraTS19_TCIA12_466_1', 'LGG__BraTS19_TCIA13_615_1', 'LGG__BraTS19_TCIA13_630_1']

i = 0

for case_ in tqdm(train_list):
    print(case_)
    
    name = []
    print(f'name：{case_}')
    gg = case_.split("__")[0]
    case = case_.split("__")[1]


    i+=1
    print(i)
    
    t1_path = os.path.join(data_root,gg, case,case+'_t1.nii.gz')
    t1ce_path = os.path.join(data_root,gg,  case,case+'_t1ce.nii.gz')
    t2_path = os.path.join(data_root,gg,  case,case+'_t2.nii.gz')
    flair_path = os.path.join(data_root,gg, case,case+'_flair.nii.gz')

    # t1_path = os.path.join(data_root,case_.split("000")[-2]+"0000.nii.gz")
    # t1ce_path = os.path.join(data_root,case_.split("000")[-2]+"0001.nii.gz")
    # t2_path = os.path.join(data_root,case_.split("000")[-2]+"0002.nii.gz")
    # flair_path = os.path.join(data_root,case_.split("000")[-2]+"0003.nii.gz")

    msk_path = os.path.join(data_root,gg, case,case+'_seg.nii.gz')

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
            '/storage/homefs/zk22e821/Dataset/Brats19_nn_training/1/{}_slice_{}.h5'.format(i, slice_ind), 'w')
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
