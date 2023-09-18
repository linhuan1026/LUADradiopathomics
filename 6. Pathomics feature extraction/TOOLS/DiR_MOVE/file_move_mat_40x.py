import os
import shutil

#################### 从一堆mat文件中筛选出想要的mat文件 ####################


# get name
patch_path='/media/gzzstation/14TB/PyRadiomics_Data/XY2/patch_40x'

wsi_paths = sorted(os.listdir(patch_path))
length=len(wsi_paths)

target_path='/media/gzzstation/14TB/PyRadiomics_Data/XY2/mat_40x'

# abs_path = '/home/gzzstation/图片/tmp/patch/'
#     # os.makedirs(save_dir, exist_ok=True)

wsi_id_list=[]

for wsi_path in wsi_paths:

    pid = wsi_path
    wsi_id_list.append(pid)

    abs_pid=patch_path+'/'+pid
    img_patch = sorted(os.listdir(abs_pid))

    for i in img_patch:
        abs_img_patch=patch_path+'/'+pid+'/'+i
        
        move_tar_path=target_path+'/'+pid+'/'+i

        shutil.copyfile(abs_img_patch, move_tar_path)

    print("########### current image move successfully ! #############")
        
print("########################")







