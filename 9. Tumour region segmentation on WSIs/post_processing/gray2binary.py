import os
import glob
import cv2
import openslide
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np
import pdb

WSI_PATH = './wsi/
GRAY_PATH = './gray_path/'
MASK_PATH = './seg_path/'
level = 3


def run():
    wsi_paths=glob.glob(os.path.join(WSI_PATH,  '*.svs'))
    wsi_paths.sort()
    pbar = tqdm(wsi_paths)
    
    for wsi_path in pbar:
        id_name = wsi_path.split('/')[-1]
        id_name = id_name.split('.svs')[0]
        # print(id_name)
        pbar.set_description("Processing %s" % id_name)

        slide = openslide.OpenSlide(wsi_path)
        size = slide.level_dimensions[level]
        w1 = size[0]
        h1 = size[1]
        # pdb.set_trace()
        gray_path = GRAY_PATH + id_name + '_gray.png'
        img = cv2.imread(gray_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        
        # plt.hist(img.ravel(), 256)
        ret1, th1 = cv2.threshold(img, 151, 255, cv2.THRESH_BINARY)       

        # pdb.set_trace()
        # ret1, th1 = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)      #OTSU
        im = cv2.resize(th1, (w1, h1), cv2.INTER_NEAREST)
     
        # plt.imshow(im, "gray")
        # plt.axis('off')
        cv2.imwrite(MASK_PATH + id_name +  '_seg.png', im)
        # print(str(ret1))
        # plt.show()
        
if __name__=="__main__":
    run()        