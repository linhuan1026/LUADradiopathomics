import os
import shutil

#################### 从一堆mat文件中按文件名字 给mat文件分类，一个病例一个文件夹存放 ####################

mat_path='/media/gzzstation/14TB/PyRadiomics_Data/XY2-2/mat_40x'

# get name
name_path='/media/gzzstation/14TB/PyRadiomics_Data/XY2-2/patch_40x'
wsi_paths = sorted(os.listdir(name_path))
length=len(wsi_paths)

target_path='/media/gzzstation/14TB/PyRadiomics_Data/XY2-2/mat_infile'

count_x=0
com_dir = mat_path
target_dir = target_path
files = os.listdir(com_dir)
# [os.path.join(com_dir,file) for file in files if file.startswith("1607862-17")]
for i in range(len(wsi_paths)):
    wsi_patches_dir = [os.path.join(com_dir,file) for file in files if file.startswith(wsi_paths[i])]
    target_dir_create = os.path.join(target_dir, wsi_paths[i])
    if os.path.exists(target_dir_create) is False:
        os.makedirs(target_dir_create)

    [shutil.copy(wsi_patches_dir1, target_dir_create) for wsi_patches_dir1 in wsi_patches_dir]
    count_x+=1
    print(count_x)
    print("######################################")
