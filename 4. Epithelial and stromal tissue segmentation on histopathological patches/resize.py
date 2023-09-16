import cv2
import os

patch_path='/media/hero/Windows 10/patch'
out_path = '/media/hero/Windows 10/patch_10x'
for i in os.listdir(patch_path):
    i_path = os.path.join(patch_path,i)
    img=cv2.imread(i_path)
    img = cv2.resize(img,(512,512))
    cv2.imwrite(os.path.join(out_path,i),img)