#!/usr/bin/env python

from __future__ import print_function

import logging
import os
import glob

import sys
sys.path.append(r'/home/gzzstation/下载/real_pathomics/real_pathomics-all-work')

import pathomics
from pathomics import featureextractor
import pandas as pd

import multiprocessing
from multiprocessing import Pool, Manager
from itertools import repeat
from pathomics.helper import save_mat_mask, save_matfiles_info_to_df, select_image_mask_from_src, save_results_to_pandas
from pathomics.helper.bmask import files_convert_color2bmask
from pathomics.helper.filter import filter_workflow
from pathomics.helper.merge_color_mask import merge_color_task

# Get the Pypathomics logger (default log-level = INFO)
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

list_error=[]

def run(imageName, maskName, features):
    # Initialize feature extractor
    extractor = featureextractor.PathomicsFeatureExtractor(**settings)

    # Disable all classes except histoqc
    extractor.disableAllFeatures()

    # Only enable morph in nuclei
    extractor.enableFeaturesByName(**features)

    featureVector = extractor.execute(imageName, maskName)
    return featureVector


if __name__ == '__main__':
    # Step 1: set enabled features and basic parameters
    features = dict(firstorder=[], glcm=[], glrlm=[])
    n_workers = 20

    BASE_PATH='/media/gzzstation/14TB/PyRadiomics_Data/SXCH/SX_10x'
    wsi_paths1 = os.listdir(BASE_PATH+'/')
    wsi_paths=sorted(wsi_paths1)
    length=len(wsi_paths)

    save_dir1 = '/media/gzzstation/14TB/PyRadiomics_Data/SXCH/save_example4'
    os.makedirs(save_dir1, exist_ok=True)

    save_csv= save_dir1+'/SX_10x_Stroma_feat_csv'
    os.makedirs(save_csv, exist_ok=True)

    for wsi_path in wsi_paths:

        name = wsi_path.split('/')[-1]
        # pid = name[:-9]
        print('name:',name)


        patient_id = name
        base_dir = BASE_PATH+'/'+patient_id
        print('base_dir:', base_dir)
        
        image_dir = f'{base_dir}/10x'
        image_suffix = '_10x.png'
        # save_dir = f'{base_dir}/{patient_id}'
        save_dir = save_dir1+'/'+patient_id
        os.makedirs(save_dir, exist_ok=True)

        # Step 2: merge color mask
        color_mask_dir = image_dir
        color_mask_suffix = '_10x-mask.png'
        save_merged_color_mask_dir = f'{save_dir}/10x_mask_merge_stroma'
        save_suffix = color_mask_suffix
        src_color = [255, 165, 0]
        tar_color = [65, 105, 225]
        merge_color_task(color_mask_dir, color_mask_suffix,
                        save_merged_color_mask_dir, save_suffix, src_color,
                        tar_color, n_workers)
        mask_dir = save_merged_color_mask_dir
        mask_suffix = color_mask_suffix

        # Step 3: epi ratio control
        color_dict = {
            'Stroma': [255, 165, 0],
            'Epi': [205, 51, 51],
            'Necrosis': [0, 255, 0],
            'Background': [255, 255, 255]
        }
        img_files = glob.glob(f'{image_dir}/*{image_suffix}')
        mask_files = glob.glob(f'{mask_dir}/*{mask_suffix}')
        select_size = 300
        print("############################")

        # aiming at the samples do not have the area of Stroma > 90%:
        #   choose the patches with the area of Stroma > 10%
        
        
        # if number of patch <= 300:  n(n=100)
        #   then all patches should be used
        #   then  choose all patches which the area of Stroma > 10%

        # if number of patch > 300: 
        #   then randomly choose 300 patches which the area of Stroma > 90%
        

        filter = dict(type='tissue_ratio_control',
                        img_files=img_files,
                        color_mask_files=mask_files,
                        color_dict=color_dict,
                        threshold=.1,
                        select_key='Stroma',
                        ignores='Background')
        img_files, color_mask_files = filter_workflow(filter,
                                                        size_control=select_size)


############################## 
        if len(color_mask_files)==0:
            list_error.append(patient_id)
            list_error_csv_path = f'{save_csv}/list_error.csv'
            df2=save_results_to_pandas(list_error, list_error_csv_path)
            print("############# this sample has no patches with Stroma area more than 90%, and has been listed ! #############")  
            continue
############################## 




        # Step 4: convert color masks into binary masks
        save_binary_mask_dir = f'{save_dir}/10x_bmask'
        mask_files = files_convert_color2bmask(color_mask_files,
                                            save_binary_mask_dir,
                                            color_dict['Stroma'])
        # Step 5: compute features
        with Pool(processes=n_workers) as pool:
            featureVectors = pool.starmap(run, zip(img_files, mask_files, repeat(features, len(img_files))))

        # Step 6: save results
        save_results_csv_path = f'{save_csv}/{patient_id}_10x_stroma_feat.csv'
        data = {}
        data['patient_id'] = []
        data['patch_id'] = []
        for i in range(len(featureVectors)):
            data['patient_id'].append(patient_id)
            img_path = img_files[i]
            name = os.path.basename(img_path)
            ext = name.split('.')[-1]
            name = name.replace(f'.{ext}', '')
            data['patch_id'].append(name)
            for feature_name, feature_value in featureVectors[i].items():
                if data.get(feature_name) is None:
                    data[feature_name] = [feature_value]
                else:
                    data[feature_name].append(feature_value)

        df = save_results_to_pandas(data, save_results_csv_path)

        
