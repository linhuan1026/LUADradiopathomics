import os
import sys
import glob
from PIL import Image
import numpy as np
from multiprocessing import Process
from multiprocessing import Pool, Manager
from itertools import repeat


def merge_color(mask_file, src_color, tar_color, save_path=None):
    mask = Image.open(mask_file)
    mask = np.array(mask)
    mask.astype(np.uint8)
    src_color = np.array(src_color)
    tar_color = np.array(tar_color)
    mask = np.where(mask == tar_color, src_color, mask)
    mask = mask.astype(np.uint8)
    if save_path == None:
        return mask
    else:
        im = Image.fromarray(mask)
        im.save(save_path)


def merge_color_task(src_dir, src_suffix, save_dir, save_suffix, src_color,
                     tar_color, n_workers):
    src_files = glob.glob(f'{src_dir}/*{src_suffix}')
    os.makedirs(save_dir, exist_ok=True)
    save_paths = []
    for src_file in src_files:
        save_path = src_file.replace(src_dir, save_dir)
        save_path = save_path.replace(src_suffix, save_suffix)
        save_paths.append(save_path)

    n = len(src_files)
    with Pool(processes=n_workers) as pool:
        pool.starmap(
            merge_color,
            zip(src_files, repeat(src_color, n), repeat(tar_color, n),
                save_paths))
