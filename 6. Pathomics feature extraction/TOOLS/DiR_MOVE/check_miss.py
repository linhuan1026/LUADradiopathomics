import pandas as pd
import os

name_select1=pd.read_csv('./GDPH_ori_select.csv')
tmp=name_select1.iloc[:,0] 
GDPH_ori_194=tmp.to_list()

patch_path='/media/gzzstation/14TB/PyRadiomics_Data/GDPH_ori/patch_40x'
wsi_paths = sorted(os.listdir(patch_path))

# print("#####################")
list_GDPH_193=[]

for wsi_path in wsi_paths:
    a=wsi_path.split('-')[0]
    list_GDPH_193.append(a)

for i in range(0, len(GDPH_ori_194)):
    if list_GDPH_193[i] in GDPH_ori_194:
        print(list_GDPH_193[i])


print(list_GDPH_193)
print(len(list_GDPH_193))
test=pd.DataFrame(data=list_GDPH_193)
test.to_csv('./list_GDPH_193.csv')

print("######### end ############")











