import os
import glob
import numpy as np
from PIL import Image
import pandas as pd
from pathomics.helper.preprocessing import multiRunStarmap


def color2bmask(img, color):
    #print(img, color)
    if isinstance(img, str):
        img = Image.open(img)
        img = np.array(img)
    #color = np.array(color)
    #print(img.shape, color.shape)
    #print(img)
    #print(np.max(img))
    mask = np.all(img == color, axis=2)
    #print(np.sum(mask))
    mask = mask.astype(np.uint8)
    return mask


def save_color2bmask(img, color, save_path):
    #print(color, color.__class__)
    mask = color2bmask(img, color)
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    if np.max(mask) != 255:
        mask = mask * 255
        mask = mask.astype(np.uint8)
    mask = Image.fromarray(mask)
    mask.save(save_path)


def src_convert_color2bmask(src_dir, src_suffix, save_dir, save_suffix, color):
    src_paths = glob.glob(f'{src_dir}{src_suffix}')
    save_paths = [
        src_path.replace(src_dir, save_dir).replace(src_suffix, save_suffix)
        for src_path in src_paths
    ]
    multiRunStarmap(save_color2bmask, src_paths, color, save_paths)
    return save_paths


def files_convert_color2bmask(src_paths, save_dir, color):
    color = np.array(color)
    src_dir = os.path.dirname(src_paths[0])
    save_paths = [
        src_path.replace(src_dir, save_dir) for src_path in src_paths
    ]
    multiRunStarmap(save_color2bmask, src_paths, color, save_paths)
    return save_paths


if __name__ == '__main__':
    csv_file = '../pathomics_prepare/example4.csv'
    color_mask_column_name = 'mask_path'
    b_mask_column_name = 'bmask_path'
    df = pd.read_csv(csv_file)
    color_mask_paths = list(df[color_mask_column_name])
    save_dir = '../pathomics_prepare/1319263-8/10x_bmask_stroma'
    src_dir = os.path.dirname(color_mask_paths[0])
    color = [255, 165, 0]
    color = np.array(color)
    save_paths = [
        src_path.replace(src_dir, save_dir) for src_path in color_mask_paths
    ]
    multiRunStarmap(save_color2bmask, color_mask_paths, color, save_paths)
    df[b_mask_column_name] = save_paths
    df.to_csv(csv_file, index=False)
