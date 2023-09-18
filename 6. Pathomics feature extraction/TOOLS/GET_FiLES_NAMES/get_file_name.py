import os
import shutil
import pandas as pd

# get name
patch_path='/media/gzzstation/14TB/PyRadiomics_Data/XY2-2/patch_40x'

list_name=[]

wsi_paths = sorted(os.listdir(patch_path))
length=len(wsi_paths)

for i in range(0, length):
    # aa=wsi_paths[i][:-9]
    aa=wsi_paths[i]
    aa=aa.split('-')[0]
    list_name.append(aa)

test=pd.DataFrame(data=list_name)
test.to_csv('./list_XY2-23.csv', header=None, index=None)

print("success!")