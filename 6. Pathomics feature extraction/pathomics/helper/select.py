import os
import sys
import glob
import pandas as pd
import numpy as np


def select_image_mask_from_src(names,
                               image_dir,
                               image_suffix,
                               mask_dir,
                               mask_suffix,
                               save_path=None):
    image_paths = []
    mask_paths = [] 
    for name in names:
        # image_path = f'{image_dir}/{name}{image_suffix}' # mat的结尾是.mat
        image_path = f'{image_dir}/{name}{image_suffix}'# mat的结尾是_new.mat
        print("############### image_path:", image_path)
        mask_path = f'{mask_dir}/{name}{mask_suffix}'
        print("############### mask_path:", mask_path)

        image_paths.append(image_path)
        mask_paths.append(mask_path)
    if save_path != None:
        df = pd.DataFrame(
            data=dict(imagepaths=image_paths, maskpaths=mask_paths))
        df.to_csv(save_path, index=False)
    return image_paths, mask_paths


class SelectionRules():

    @staticmethod
    def top(data, size, key_name):
        """
        input:
            data: pandas data frame
            size: select size
            key_name: based on which colomn, specifiy by key_name
        return: patient_ids depends on top size of data[key_name]
        """
        a = data[key_name]
        l = list(a)
        rank = [
            index
            for index, value in sorted(list(enumerate(l)), key=lambda x: x[1])
        ]
        if size > len(rank):
            size = len(rank)
        sel = []
        for i in rank:
            sel.append(data['patch_id'][i])
        return sel

    @staticmethod
    def random(size, select_size):
        pass


def select(image_dir, image_suffix, mask_dir, mask_suffix, info_csv_file,
           select_rule, select_size, select_key_name, save_path):
    data = pd.read_csv(info_csv_file)
    select_fun = getattr(SelectionRules, select_rule)
    sel_patch_ids = select_fun(data, select_size, select_key_name)
    sel_data = {'patch_id': [], 'image_path': [], 'mask_path': []}
    for patch_id in sel_patch_ids:
        sel_data['patch_id'].append(patch_id)
        sel_data['image_path'].append(
            f'{image_dir}{os.sep}{patch_id}{image_suffix}')
        sel_data['mask_path'].append(
            f'{mask_dir}{os.sep}{patch_id}{mask_suffix}')
    df = pd.DataFrame(data=sel_data)
    if save_path != None:
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(save_path, index=False)
    else:
        return df


if __name__ == '__main__':
    image_dir = '../pathomics_prepare/1319263-8/40x'
    image_suffix = '.png'
    mask_dir = '../pathomics_prepare/1319263-8/mask'
    mask_suffix = '.png'
    info_csv_file = '../pathomics_prepare/1319263-8_inst_uid_info.csv'
    select_rule = 'top'
    select_size = 400
    select_key_name = 'inst_uid_size'
    save_path = '../pathomics_prepare/1319263-8.csv'
    select(image_dir, image_suffix, mask_dir, mask_suffix, info_csv_file,
           select_rule, select_size, select_key_name, save_path)