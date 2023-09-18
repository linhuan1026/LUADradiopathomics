#!/usr/bin/env python

from __future__ import print_function
import os
import logging

import glob
import os

import sys
sys.path.append(r'/home/gzzstation/下载/real_pathomics/real_pathomics-all-work')

import pathomics
from pathomics import featureextractor
import pandas as pd
from PIL import Image
import numpy as np

import multiprocessing
from multiprocessing import Pool, Manager
from itertools import repeat
from pathomics.helper import save_mat_mask, save_matfiles_info_to_df, select_image_mask_from_src, save_results_to_pandas
from pathomics.helper.preprocessing import get_low_magnification_WSI
from pathomics.helper.preprocessing import get_low_magnification_WSI_new

# Get the Pypathomics logger (default log-level = INFO)##################### length: 283
logger = pathomics.logger
logger.setLevel(
    logging.DEBUG
)  # set level to DEBUG to include debug log messages in log file


# Set up the handler to write out all log entries to a file
handler = logging.FileHandler(filename='testLog.txt', mode='w')
formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


# Define settings for signature calculation
# These are currently set equal to the respective default values
settings = {}

count_x=0
def run(imageName, maskName, features):

    # Initialize feature extractor
    extractor = featureextractor.PathomicsFeatureExtractor(**settings)  #### 这一句出问题 运行太慢

    # Disable all classes except histoqc
    extractor.disableAllFeatures()
    print("######################### extractor before:", extractor)


    # Only enable morph in nuclei
    extractor.enableFeaturesByName(**features) 
    print("######################### extractor after:", extractor)

    featureVector = extractor.execute(imageName, maskName)
    return featureVector


if __name__ == '__main__':

    # Step 1: set enabled features and basic parameters，  设置允许的特征 和 基本参数
    features = dict(firstorder=[], glcm=[], glrlm=[])
    print("######################### features:", features)
    n_workers = 10


    base_dir = '/media/gzzstation/14TB/PyRadiomics_Data/XY2-2/WSI_2.5x'
    dir_target = '/media/gzzstation/14TB/PyRadiomics_Data/XY2-2/mask_2.5x'
    save_dir='/media/gzzstation/14TB/PyRadiomics_Data/XY2-2/XY2-2_save_example2'

    wsi_paths1 = glob.glob(os.path.join(base_dir+'/', '*.svs'))           
    wsi_paths=sorted(wsi_paths1)
    length=len(wsi_paths)

    for wsi_path in wsi_paths:

        count_x=count_x+1
        print("##################### length:", length)
        print("##################### count_x:", count_x)
        # import pdb
        # pdb.set_trace()
        # obtain the wsi path
        name = wsi_path.split('/')[-1]
        pid = name[:-4]
        print('name:',name)
        print('pid:', pid)

        image_path=f'{base_dir}/{name}'
        patient_id=name[:-4]
        mask_path=f'{dir_target}/{patient_id}_mask.png'
        save_dir=f'{save_dir}'

        # Step 2: get low magnification image file，    降采样图片，40倍到2.5倍
        low_mag_wsi_save_path = f'{save_dir}/{patient_id}.png'
        max_magnification = 40
        save_magnification = 2.5
        get_low_magnification_WSI(image_path, max_magnification, save_magnification, low_mag_wsi_save_path)
        # get_low_magnification_WSI_new(image_path, max_magnification, save_magnification, low_mag_wsi_save_path)
        
        image_path = low_mag_wsi_save_path

        # Step 4: compute features by multiprocessing
        # can load as numpy array
        image = Image.open(image_path)
        image = np.array(image)

        mask = Image.open(mask_path)    
        # Plu-170003 TP - 2022-01-12 22.30.45
        # Plu-170003 TP - 2022-01-12 22.30.45_mask
        mask = np.array(mask) 

        image_path = image
        mask_path = mask    
        
        featureVector = run(image_path, mask_path, features)
        print("############### featureVector", featureVector)

        # Step 5: save results
        save_results_csv_path = f'{save_dir}/{patient_id}_2.5x_tumorbed_feat.csv'

        data = {}
        data['patient_id'] = patient_id
        for feature_name, feature_value in featureVector.items():
            if data.get(feature_name) is None:
                data[feature_name] = [feature_value]
                print("data.get(feature_name) is none")
            else:
                data[feature_name].append(feature_value)
                print("data.get(feature_name) is NOT none")

        df = save_results_to_pandas(data, save_results_csv_path)
