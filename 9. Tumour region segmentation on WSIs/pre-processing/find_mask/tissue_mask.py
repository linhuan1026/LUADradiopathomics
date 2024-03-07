import sys
import os
import argparse
import logging
import pdb
import numpy as np
import openslide
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

parser = argparse.ArgumentParser(description='Get tissue mask of WSI and save'
                                 ' it in npy format')
'''parser.add_argument('--wsi_path', default='./wsi', metavar='WSI_PATH', type=str,
                    help='Path to the WSI file')'''
'''parser.add_argument('--npy_path', default='./patient_002.npy', metavar='NPY_PATH', type=str,
                    help='Path to the output npy mask file')'''
parser.add_argument('--level', default=  3 , type=int, help='at which WSI level'
                    ' to obtain the mask, default 6')
parser.add_argument('--RGB_min', default=0, type=int, help='min value for RGB'
                    ' channel, default 50')


def run(args,wsi_wholepath,out_tissue_mark):
    logging.basicConfig(level=logging.INFO)

    slide = openslide.OpenSlide(wsi_wholepath)
    
    # note the shape of img_RGB is the transpose of slide.level_dimensions
    
    img_RGB = np.transpose(np.array(slide.read_region((0, 0),args.level,
                           slide.level_dimensions[args.level]).convert('RGB')),
                           axes=[1, 0, 2])
    
    
    # print(img_RGB)
    img_HSV = rgb2hsv(img_RGB)
    # pdb.set_trace()
    # background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
    # background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
    # background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
    background_R = img_RGB[:, :, 0] > 255
    background_G = img_RGB[:, :, 1] > 255
    background_B = img_RGB[:, :, 2] > 255
    tissue_RGB = np.logical_not(background_R & background_G & background_B)
    tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
    tissue_H = img_HSV[:, :, 0] > threshold_otsu(img_HSV[:, :, 0])
    tissue_V = img_HSV[:, :, 2 ] > threshold_otsu(img_HSV[:, :, 2])
    min_R = img_RGB[:, :, 0] > args.RGB_min
    min_G = img_RGB[:, :, 1] > args.RGB_min
    min_B = img_RGB[:, :, 2] > args.RGB_min
    
    tissue_mask = tissue_S & tissue_H & tissue_V & tissue_RGB & min_R & min_G & min_B


    np.save(out_tissue_mark, tissue_mask)
    
    plt.ion()          
    plt.imshow(tissue_mask, cmap = plt.get_cmap('gray'))
    plt.axis('off')
    print(tissue_mask.shape)
    plt.subplots_adjust(top = 1, bottom =0,  left = 0, right = 1, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.savefig(out_tissue_mark + '_gray.png', bbox_inches='tight',pad_inches=0.0)
    # plt.colorbar()
    # plt.clim(0.00, 1.00)
    # plt.axis([0, tissue_mask.shape[1], 0, tissue_mask.shape[0]])
    # plt.savefig(out_tissue_mark + '.png')
    plt.show()
    plt.pause(1
    plt.close()         



def main():
    args = parser.parse_args()
    wsi_path = './wsi/'
    for wsi in os.listdir(wsi_path):
        wsi_wholepath = os.path.join(wsi_path,wsi)
        (wsi_path, wsi_extname) = os.path.split(wsi_wholepath)
        (wsi_name,extension) = os.path.splitext(wsi_extname)
        out_tissue_mark = './tissue_mask/%s.npy'%wsi_name
        if os.path.exists(out_tissue_mark):
            continue
        else:
            print(wsi_name)
        run(args,wsi_wholepath,out_tissue_mark)
        print("finish,{}\n".format(wsi_name))

if __name__ == '__main__':
    main()