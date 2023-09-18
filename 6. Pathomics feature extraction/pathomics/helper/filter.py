from http.client import REQUESTED_RANGE_NOT_SATISFIABLE
import os
import sys
import glob
import pandas as pd
from PIL import Image
import numpy as np


class Filter():

    @staticmethod
    def tissue_ratio_control(img_files,
                             color_mask_files,
                             color_dict,
                             select_key,
                             threshold,
                             ignores=None):
        """
        filter files depend on the ratio/threshold of select_dict according to color_dict,
        note img_file and color_mask_file have the same order
        """

        def match_ratio(color_mask_file, color_dict, select_key, threshold,
                        ignores):
            color_mask = Image.open(color_mask_file)
            np_color_mask = np.array(color_mask)
            content = {}
            if ignores != None:
                if isinstance(ignores, str):
                    ignores = [ignores]
            for k, v in color_dict.items():
                if k not in ignores:
                    a = np.sum(np.all(np_color_mask == np.array(v), axis=2))
                    #print(k, a)
                    content[k] = a
            sum = 0
            for k, v in content.items():
                sum += v
            #print(content)
            #print(content[select_key], sum)
            ratio = content[select_key] / sum

            #print(ratio)
            #print(content)
            if ratio >= threshold:
                #print(ratio)
                return True
            else:
                return False

        res_img_files = []
        res_color_mask_files = []
        for i in range(len(img_files)):
            img_file = img_files[i]
            #print(img_file)
            color_mask_file = color_mask_files[i]
            if match_ratio(color_mask_file, color_dict, select_key, threshold,
                           ignores):
                res_img_files.append(img_file)
                res_color_mask_files.append(color_mask_file)
            #break
        return res_img_files, res_color_mask_files


def filter_workflow(filter, size_control):
    """
    input:
        filters: [<filter>], filter: dict of filter parameters
        relations: str, '-', '+', fusion results of different filters
        size_control: final size needed of filtered files
    return:
        image_files, mask_files 
    """

    filter_type = filter.get('type')
    filter_fun = getattr(Filter, filter_type)
    img_files, mask_files = filter_fun(filter.get('img_files'),
                                       filter.get('color_mask_files'),
                                       filter.get('color_dict'),
                                       filter.get('select_key'),
                                       filter.get('threshold'),
                                       filter.get('ignores'))
    return img_files[:size_control], mask_files[:size_control]


if __name__ == '__main__':
    img_dir = '../pathomics_prepare/1319263-8/10x'
    mask_dir = '../pathomics_prepare/1319263-8/10x_mask_merge_stroma'
    img_suffix = '_10x.png'
    mask_suffix = '_10x-mask.png'
    color_dict = {
        'Stroma': [255, 165, 0],
        'Epi': [205, 51, 51],
        'Necrosis': [0, 255, 0],
        'Background': [255, 255, 255]
    }
    img_files = glob.glob(f'{img_dir}/*{img_suffix}')
    mask_files = glob.glob(f'{mask_dir}/*{mask_suffix}')
    filter1 = dict(type='tissue_ratio_control',
                   img_files=img_files,
                   color_mask_files=mask_files,
                   color_dict=color_dict,
                   threshold=.7,
                   select_key='Epi',
                   ignores='Background')
    filter2 = dict(type='tissue_ratio_control',
                   img_files=img_files,
                   color_mask_files=mask_files,
                   color_dict=color_dict,
                   threshold=.9,
                   select_key='Stroma',
                   ignores='Background')

    img_files, mask_files = filter_workflow(filter1, size_control=300)
    print(len(img_files))
    data = {'image_path': img_files, 'mask_path': mask_files}
    df = pd.DataFrame(data=data)
    df.to_csv('../pathomics_prepare/example3.csv', index=False)

    img_files, mask_files = filter_workflow(filter2, size_control=300)
    print(len(img_files))
    data = {'image_path': img_files, 'mask_path': mask_files}
    df = pd.DataFrame(data=data)
    df.to_csv('../pathomics_prepare/example4.csv', index=False)
