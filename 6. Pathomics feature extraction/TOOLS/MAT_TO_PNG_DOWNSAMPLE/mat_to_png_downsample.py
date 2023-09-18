from scipy.io import loadmat
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
from PIL import Image

import glob
import os

# patch_dir='/media/gzzstation/14TB/PyRadiomics_Data/SXCH/patch_40x/'
# base_dir = '/media/gzzstation/14TB/PyRadiomics_Data/SXCH/mat_40x/'

# wsi_paths1 = os.listdir(base_dir+'/')
# wsi_paths=sorted(wsi_paths1)
# length=len(wsi_paths)

# save_dir = '/media/gzzstation/14TB/PyRadiomics_Data/SXCH/SX_10x/'
# os.makedirs(save_dir, exist_ok=True)

###########################################

patch_dir='/media/gzzstation/14TB/PyRadiomics_Data/XY2/patch_40x/'
base_dir = '/media/gzzstation/14TB/PyRadiomics_Data/XY2/mat_40x/'

wsi_paths1 = os.listdir(patch_dir+'/')
wsi_paths=sorted(wsi_paths1)
length=len(wsi_paths)

save_dir = '/media/gzzstation/14TB/PyRadiomics_Data/XY2/XY2_10x/'
os.makedirs(save_dir, exist_ok=True)

count_x=0
for wsi_path in wsi_paths:
    # obtain the wsi path
    name = wsi_path.split('/')[-1]
    # pid = name[:-9]
    print('name:',name)

    patient_id = name
    image_dir = f'{base_dir}/{patient_id}'
    image_suffix = '.png'

    patch_id1=os.listdir(image_dir+'/')
    patch_id=sorted(patch_id1)
    # for patch_id in 

    for img_patch in patch_id:

        name_patch = img_patch.split('/')[-1]
        pid_patch = name_patch[:-4]
        # print(name)
        # print('pid_patch:', pid_patch)

        mat=loadmat(image_dir+'/'+name_patch)

        png=mat['tissue_map'] # 这句报错：no tissue_map
        png2=Image.fromarray(png)
        png3=png2.convert('P')

        # png3.putpalette([205, 51, 51,255, 165, 0 ,0, 255, 0,255,255,255])
        png3.putpalette([
            205, 51, 51,    #0 红 肿瘤
            255, 165, 0 ,   #1 黄 间质
            65, 105, 225,   #2 蓝 淋巴
            0, 255, 0,      #3 绿 坏死
            255,255,255     #4 白 背景
            ])
# 拯云的：肿瘤 间质 淋巴 坏死 背景
# 泰哥的： 肿瘤 坏死 淋巴 间质 背景

        png4=png3.convert('RGB')
        png4 = png4.resize((512,512)) # 这个是mask的图像，下采样四倍
        
        
        # 读取40x下的he patch图像
        try:
            he_img = Image.open(patch_dir+patient_id+'/'+pid_patch+'.png').convert('RGB')
            he_img = he_img.resize((512,512))


            if not os.path.exists(save_dir+patient_id+'/10x'):
                os.makedirs(save_dir+patient_id+'/10x')

            he_img.save(os.path.join(save_dir+patient_id+'/10x/'+str(pid_patch[:-4])+'_10x.png'))
            png4.save(os.path.join(save_dir+patient_id+'/10x/'+str(pid_patch[:-4])+'_10x-mask.png'))
        
        except Exception as e:
            he_img = Image.open(patch_dir+patient_id+'/'+pid_patch[:-4]+'.png').convert('RGB')
            he_img = he_img.resize((512,512))


            if not os.path.exists(save_dir+patient_id+'/10x'):
                os.makedirs(save_dir+patient_id+'/10x')

            he_img.save(os.path.join(save_dir+patient_id+'/10x/'+str(pid_patch[:-8])+'_10x.png'))
            png4.save(os.path.join(save_dir+patient_id+'/10x/'+str(pid_patch[:-8])+'_10x-mask.png'))



    count_x+=1
    print(count_x)
    print("save successfully!")
