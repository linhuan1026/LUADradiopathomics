import os
import glob
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt 
import numpy as np
import pdb

NPY_PATH = './npy_path/'
PNG_PATH = './png_path/'

def run():
    npy_paths=glob.glob(os.path.join(NPY_PATH, '*.npy'))
    npy_paths.sort()
    pbar = tqdm(npy_paths)
    
    for npy in pbar:
        
        id_name = npy.split('/')[-1]
        id_name = id_name.split('.npy')[0]
        # print(id_name)
        pbar.set_description("Processing %s" % id_name)
   
        img = np.load(npy, allow_pickle=True)
        img = np.transpose(img)

        plt.imshow(img, cmap = plt.get_cmap('gray'))        #gray_image
        plt.axis('off')
        plt.subplots_adjust(top = 1, bottom =0,  left = 0, right = 1, hspace = 0, wspace = 0)
        plt.margins(0,0)
        # plt.savefig(PNG_PATH + id_name + '_gray.png')         
        plt.savefig(PNG_PATH + id_name + '_gray.png', bbox_inches='tight',pad_inches=0.0)
        # plt.pause(1)   
        plt.close()         

        # plt.imshow(img, cmap = plt.get_cmap('jet'))       #heatmap
        # plt.colorbar()
        # plt.savefig(PNG_PATH + id_name + '_jet.png')
        # plt.show()  


if __name__=="__main__":
    run()
