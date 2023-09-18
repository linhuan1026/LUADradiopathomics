import os
import shutil
import pandas as pd
import numpy as np

#################### 按需求，筛选出所需的patch文件和mat文件 ####################

# get name
# patch_path='/media/gzzstation/Zhengze/JMCH/patch_40x'
# wsi_paths = sorted(os.listdir(patch_path))
# length=len(wsi_paths)

list_not_exist=[]

name_select1=pd.read_csv('./patch_select.csv')
tmp=name_select1.iloc[:,0]
wsi_paths=tmp.to_list()

ori_patch_path='/media/gzzstation/14TB/PyRadiomics_Data/XY2/patch_40x'
ori_mat_path=  '/media/gzzstation/14TB/PyRadiomics_Data/XY2/mat_40x'

ori_patch_name=sorted(os.listdir(ori_patch_path))

tar_patch_path='/media/gzzstation/14TB/PyRadiomics_Data/XY2/patch_select'
tar_mat_path='/media/gzzstation/14TB/PyRadiomics_Data/XY2/mat_select'

# abs_path = '/home/gzzstation/图片/tmp/patch/'
#     # os.makedirs(save_dir, exist_ok=True)

wsi_id_list=[]

for wsi_path in wsi_paths:

    pid = wsi_path
    wsi_id_list.append(pid)

    try:
    # move patch
        pid=[file for file in ori_patch_name if file.startswith(str(pid))]
        pid=pid[0]

        patch_from=ori_patch_path+'/'+str(pid)
        patch_to=tar_patch_path+'/'+str(pid)
        shutil.move(patch_from, patch_to)
    # move mat
        mat_from=ori_mat_path+'/'+str(pid)
        mat_to=tar_mat_path+'/'+str(pid)
        shutil.move(mat_from, mat_to)

        print("########### current image move successfully ! #############")
    except Exception as e:
        list_not_exist.append(pid)
        print("########### current image not exist ! #############")
        continue

print(list_not_exist)
test=pd.DataFrame(data=list_not_exist)
test.to_csv('./list_not_exist_SX.csv')
print("########################")







