import sys
import os
import argparse
import logging

import numpy as np
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

parser = argparse.ArgumentParser(description="Get the normal region"
                                             " from tumor WSI ")
'''parser.add_argument("--tumor_path", default='./preprocessing/tumor_np', metavar='TUMOR_PATH', type=str,
                    help="Path to the tumor mask npy")
parser.add_argument("--tissue_path", default='./preprocessing/tissue_np', metavar='TISSUE_PATH', type=str,
                    help="Path to the tissue mask npy")
parser.add_argument("--normal_path", default='./preprocessing/non_tumor_np', metavar='NORMAL_PATCH', type=str,
                    help="Path to the output normal region from tumor WSI npy")'''


def run(wsi_np_whole_path,tissue_path,non_tumor_path):
    tumor_mask = np.load(wsi_np_whole_path)
    tissue_mask = np.load(tissue_path)

    non_tumor_mask = tissue_mask & (~ tumor_mask)

    np.save(non_tumor_path, non_tumor_mask)
    # plt.imshow(non_tumor_mask)
    # print(non_tumor_mask.shape)
    # plt.colorbar()
    # plt.clim(0.00, 1.00)
    # plt.axis([0, non_tumor_mask.shape[1], 0, non_tumor_mask.shape[0]])
    # plt.savefig(non_tumor_path + '.png')
    # plt.show()

def main():
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    tumor_path = ./tumor_mask'
    for wsi_np_extname in os.listdir(tumor_path):
        wsi_np_whole_path = os.path.join(tumor_path,wsi_np_extname)
        (wsi_np_path,wsi_np_extname) = os.path.split(wsi_np_whole_path)
        (wsi_np_name,extension) = os.path.splitext(wsi_np_extname)   #wsi_np_name is name
        tissue_path = './tissue_mask/%s.npy' %wsi_np_name
        # wsi_np_name = 'non_'+wsi_np_name
        non_tumor_path = './non_tumor/%s.npy'%wsi_np_name
        run(wsi_np_whole_path,tissue_path,non_tumor_path)

        print("finish,{}\n".format(wsi_np_name))

if __name__ == "__main__":
    main()