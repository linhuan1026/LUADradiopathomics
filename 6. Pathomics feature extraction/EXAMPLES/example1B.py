#!/usr/bin/env python

from __future__ import print_function
import os
import logging

import sys
sys.path.append(r'/home/gzzstation/下载/real_pathomics/real_pathomics-all-work')

import pathomics
from pathomics import featureextractor
import pandas as pd

import multiprocessing
from multiprocessing import Pool, Manager
from itertools import repeat

from pathomics.helper import save_mat_mask, save_matfiles_info_to_df, select_image_mask_from_src, save_results_to_pandas, save_mat_mask_rm_outerNuclei
import numpy as np

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


def run(imageName, maskName, features):
    # Initialize feature extractor
    extractor = featureextractor.PathomicsFeatureExtractor(**settings)

    # Disable all classes except histoqc
    extractor.disableAllFeatures()

    # Only enable morph in nuclei
    extractor.enableFeaturesByName(**features)

    try:
        featureVector = extractor.execute(imageName, maskName, ifbinarization=False)
        print(imageName, '-----------------------------------')
        return featureVector
    except Exception as e:
        print(e)
        print(imageName)
        print(maskName)
        return None   


if __name__ == '__main__':

    import time
    start = time.time()
    # Step 1: set enabled features and basic parameters
    features = dict(props=[], haralickBasis=[])
    n_workers = 20

    list_error_patch=[] 
    list_new=[]


    base_dir = '/media/gzzstation/14TB/PyRadiomics_Data/XY2'
    wsi_paths1 = os.listdir('/media/gzzstation/14TB/PyRadiomics_Data/XY2/patch_40x/')
    wsi_paths=sorted(wsi_paths1)
    length=len(wsi_paths)

    save_dir = '/media/gzzstation/14TB/PyRadiomics_Data/XY2/example_1B'
    # os.makedirs(save_dir, exist_ok=True)

    for wsi_path in wsi_paths:

        name = wsi_path.split('/')[-1]
        # pid = name[:-9]
        print('name:',name)

        patient_id = name
        image_dir = f'{base_dir}/patch_40x/{patient_id}'
        image_suffix = '.png'


        # Step 2: convert mat-format nuclear masks into png files
        # mat_dir = f'{base_dir}/mat'
        # mat_var_name = 'inst_map'
        # mat_var_name_2 = 'inst_type'
        # target_value = 1
        # png_mask_save_dir = f'{save_dir}/mask'
        # png_mask_filepaths = save_mat_mask_rm_outerNuclei(mat_dir, mat_var_name, mat_var_name_2, target_value, 
        #                                 png_mask_save_dir, n_workers)

        mat_dir = f'{base_dir}/mat_40x/{patient_id}'
        mat_var_name = 'inst_map'
        mat_var_name_2 = 'inst_type'
        target_value = 1
        png_mask_save_dir = f'{save_dir}/mask/{patient_id}'
        png_mask_filepaths = save_mat_mask_rm_outerNuclei(mat_dir, mat_var_name, mat_var_name_2, target_value, 
                                        png_mask_save_dir, n_workers)

        # Step 3: filter patches, get first 400 patches with max nuclear size
        select_size = 400
        info_name = mat_var_name_2 # mat_var_name_2 = 'inst_type'
        only_ndarray_size = True
        save_info_path = f'{save_dir}/{patient_id}_{info_name}.csv'
        df = save_matfiles_info_to_df(mat_dir, n_workers, info_name,
                                    only_ndarray_size, target_value, save_info_path)
        sort_col_name = info_name + f'_{target_value}_size'
        df = df.sort_values(by=sort_col_name, ascending=False, key=lambda col: col)
        save_results_to_pandas(df, save_info_path)
        select_names = list(df['name'])[:select_size]

        save_select_filepaths_info = f'{save_dir}/{patient_id}_top_400_nuclear_size_files.csv'
        image_paths, mask_paths = select_image_mask_from_src(
            select_names, image_dir, image_suffix, png_mask_save_dir, '.png',
            save_select_filepaths_info)

        # Step 4: compute features by multiprocessing
        with Pool(processes=n_workers) as pool:
            featureVectors = pool.starmap(
                run,
                zip(image_paths, mask_paths, repeat(features, len(image_paths))))


##############################
# step additional:
# 看featureVectors中的特定这行 shape或者size的长度 少于多少就把这行记录下来 并且从字典里面删除
        for i in range(len(featureVectors)):
            try:
                if featureVectors[i]['props_area'].size > 3:
                    list_new.append(featureVectors[i])
                else:
                    error_patch_name=image_paths[i].split('/')[-1]
                    list_error_patch.append(error_patch_name)
                    
                    save_error_path=f'{save_dir}/{patient_id}_error_patch.csv'
                    df2 = save_results_to_pandas(data, save_error_path)
            
                print("############# this patch is sick. delete. #############", error_patch_name)
            except Exception as e:
                print(e)
                print("featureVectors is None")
                continue 
            
        featureVectors=list_new
##############################


        # Step 5: save results
        import numpy as np
        save_results_csv_path = f'{save_dir}/{patient_id}_40x_single_nuclear_feat.csv'
        data = {}
        data['patient_id'] = []
        data['patch_id'] = []
        for i in range(len(featureVectors)):
            if featureVectors[i] is None:
                continue
            if i < len(select_names):
                data['patient_id'] = data['patient_id'] + [patient_id,]*len(featureVectors[i][list(featureVectors[i])[0]])
                data['patch_id'] = data['patch_id'] + [select_names[i],]*len(featureVectors[i][list(featureVectors[i])[0]])
                
                for feature_name, feature_value in featureVectors[i].items():
                    if data.get(feature_name) is None:
                        data[feature_name] = feature_value
                    else:
                        data[feature_name] = np.concatenate([data[feature_name], feature_value])

        df = save_results_to_pandas(data, save_results_csv_path)
        end = time.time()
        print('time:', (end-start)/60)
