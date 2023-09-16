import os
from tqdm import tqdm
from glob import glob
import shutil
img_path='/media/hero/ECBE3535BE34FA22/JIANGMEN/201904076-3,4-HE'
mat_path='/media/hero/Elements/JIANGMEN/seg_cell/40x/mat'
out_path='/media/hero/ECBE3535BE34FA22/JIANGMEN/201904076-3,4-HE_mat_un'

img_list=os.listdir(img_path)
for _img in tqdm(img_list):
    _mat_path = os.path.join(mat_path,_img.replace(".png",".mat"))
    if os.path.exists(_mat_path):
        shutil.copy(_mat_path,out_path)

