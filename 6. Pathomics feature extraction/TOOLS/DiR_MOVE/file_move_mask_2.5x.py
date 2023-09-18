import os
import shutil
import pandas as pd

#################### 从一堆2.5x的mask文件中挑选出想要的mask ####################


# get name
# patch_path='/media/gzzstation/14TB/PyRadiomics_Data/XYCH/patch_40x'
# # patch_path='/media/gzzstation/14TB/PyRadiomics_Data/GDPH_add/patch_40x'

# wsi_paths = sorted(os.listdir(patch_path))
# length=len(wsi_paths)
name_select1=pd.read_csv('./list_SX_114.csv')
tmp=name_select1.iloc[:,0]
wsi_paths=tmp.to_list()

########################################
# src_path='/media/gzzstation/14TB/PyRadiomics_Data/SXCH/SXCH_wsi_2.5x/'
src_path='/media/gzzstation/14TB/PyRadiomics_Data/SXCH/wsi_mask_2.5x/'

target_path='/media/gzzstation/14TB/PyRadiomics_Data/SXCH/wsi_mask_select/'
# target_path='/media/gzzstation/14TB/PyRadiomics_Data/GDPH_add/WSI_mask_2.5x/'


list_not_exist=[]

count_x=0
for wsi_path in wsi_paths:

    pid=wsi_path

    # src_pic_path=src_path+str(pid)+'.ndpi'
    src_pic_path=src_path+str(pid)+'_mask.png'
    tar_pic_path=target_path

    # if src_pic_path=='/media/gzzstation/14TB/PyRadiomics_Data/tumor_mask-GDPH/1138633-B_mask.png':
    #     continue

    if os.path.exists(src_pic_path)==False:
        a=src_pic_path.split('/')[-1]
        aa=a.split('_')[0]
        list_not_exist.append(aa)
        continue


    shutil.move(src_pic_path, tar_pic_path)
    count_x+=1
    print(count_x)
    print("########### current image move successfully ! #############")


print(list_not_exist)
test=pd.DataFrame(data=list_not_exist)
test.to_csv('./list_not_exist_SX_mask.csv')
# test.to_csv('./list_not_exist_GDPH_add2.csv')
        
print("########################")







