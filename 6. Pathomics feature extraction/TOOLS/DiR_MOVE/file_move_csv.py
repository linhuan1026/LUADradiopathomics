import os
import shutil
import pandas as pd


#################### 从一堆csv文件中挑选出想要的csv ####################


# get name
name_select1=pd.read_csv('./list_XY1_202.csv')
tmp=name_select1.iloc[:,0]
wsi_paths=tmp.to_list()

wsi_id_list=[]

########################################
src_path='/media/gzzstation/14TB/PyRadiomics_Results/example2/XY1_save_example2/XY_example2_csv/'

target_path='/media/gzzstation/14TB/PyRadiomics_Results/example2/XY1_save_example2/example2_selected/'
# target_path='/media/gzzstation/14TB/PyRadiomics_Data/GDPH_add/WSI_mask_2.5x/'

ori_patch_path='/media/gzzstation/14TB/PyRadiomics_Data/XYCH/patch_40x'
ori_patch_name=sorted(os.listdir(ori_patch_path))

list_not_exist=[]

count_x=0

for wsi_path in wsi_paths:

    pid = wsi_path
    wsi_id_list.append(pid)

# move patch
    pid=[file for file in ori_patch_name if file.startswith(str(pid))]
    pid=pid[0]

    src_pic_path=src_path+str(pid)+'_2.5x_tumorbed_feat.csv'
    tar_pic_path=target_path

    # if src_pic_path=='/media/gzzstation/14TB/PyRadiomics_Data/tumor_mask-GDPH/1138633-B_mask.png':
    #     continue

    if os.path.exists(src_pic_path)==False:
        a=src_pic_path.split('/')[-1]
        aa=a.split('_')[0]
        list_not_exist.append(aa)
        continue


    shutil.copy(src_pic_path, target_path)
    count_x+=1
    print(count_x)
    print("########### current image move successfully ! #############")


print(list_not_exist)
test=pd.DataFrame(data=list_not_exist)
test.to_csv('./list_not_exist_XY1.csv')
# test.to_csv('./list_not_exist_GDPH_add2.csv')
        
print("########################")







