import os
import sys
import logging
import argparse

import numpy as np
import openslide
import cv2
import json
from matplotlib import pyplot as plt
import pdb

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

parser = argparse.ArgumentParser(description='Get tumor mask of tumor-WSI and ''save it in npy format')
'''parser.add_argument('--wsi_path', default./WSI', metavar='WSI_PATH', type=str,
                    help='Path to the WSI file')'''
'''parser.add_argument('--json_path', default='./json', metavar='JSON_PATH', type=str,
                    help='Path to the JSON file')'''
'''parser.add_argument('--npy_path', default='./preprocessing/tumor_np/patient_001_tumor.npy', metavar='NPY_PATH', type=str,
                    help='Path to the output npy mask file')'''
parser.add_argument('--level', default=6, type=int, help='at which WSI level'
                    ' to obtain the mask, default 6')


def run(args,wsi_wholepath,json_path,out_tumor_mask):

    # get the level * dimensions e.g. tumor0.tif level 6 shape (1589, 7514)
    # pdb.set_trace()
    slide = openslide.OpenSlide(wsi_wholepath)
    width, height = slide.level_dimensions[args.level]
    mask_tumor = np.zeros((height, width)) # the init mask, and all the value
    #  is 0

    # get the factor of level * e.g. level 6 is 2^6
    factor = slide.level_downsamples[args.level]


    with open(json_path) as f:
        dicts = json.load(f)
    tumor_polygons = dicts['positive']
   
    for tumor_polygon in tumor_polygons:
        name = tumor_polygon["name"]
        print('name:',name)
        vertices = np.array(tumor_polygon["vertices"]) / factor
        vertices = vertices.astype(np.int32)

        cv2.fillPoly(mask_tumor, [vertices], (255))

    mask_tumor = mask_tumor[:] > 127
    mask_tumor = np.transpose(mask_tumor)
    np.save(out_tumor_mask, mask_tumor)
   
    plt.imshow(mask_tumor)
    print(mask_tumor.shape)
    plt.colorbar()
    plt.clim(0.00, 1.00)
    plt.axis([0,mask_tumor.shape[1], 0, mask_tumor.shape[0]])
    plt.savefig(out_tumor_mask + '.png')
    plt.show()

def main():
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    dataclass ="tumor"
    wsi_path = './wsi'
    for tumor in os.listdir(wsi_path):
        wsi_wholepath = os.path.join(wsi_path,tumor)
        (wsi_path,wsi_extname) = os.path.split(wsi_wholepath)
        (wsi_name,extension) = os.path.splitext(wsi_extname)
        json_path = './json/%s.json'%wsi_name
        print(json_path)
        out_tumor_mask ='./tumor_mask/%s.npy'%wsi_name
        run(args,wsi_wholepath,json_path,out_tumor_mask)
        print("finish,{}\n".format(wsi_name))

if __name__ == "__main__":
    main()
