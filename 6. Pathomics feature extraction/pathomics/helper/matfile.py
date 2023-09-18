import scipy.io as sio
import os
import sys
import glob
from PIL import Image
import multiprocessing
from multiprocessing import Pool, Manager
from itertools import repeat
import numpy as np

np.set_printoptions(threshold=sys.maxsize)
import pandas as pd
from .csvfile import save_results_to_pandas


def read_mat_mask(mat_file, var_name):
    mat = sio.loadmat(mat_file)
    mask = mat[var_name]
    return mask.astype(np.uint8)


def save_one_mat_mask(mat_file, var_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    mask = read_mat_mask(mat_file, var_name)
    name = os.path.basename(mat_file).replace('.mat', '')
    im = Image.fromarray(mask)
    save_path = f'{save_dir}{os.sep}{name}.png'
    im.save(save_path)
    return save_path


def save_mat_mask(src_dir, var_name, save_dir, n_workers):
    files = glob.glob(f'{src_dir}{os.sep}*.mat')
    size = len(files)
    with Pool(processes=n_workers) as pool:
        res = pool.starmap(
            save_one_mat_mask,
            zip(files, repeat(var_name, size), repeat(save_dir, size)))
    return res

def read_mat_mask_if_only_tumor(mat_file, var_name, var_name_extend1=None, target_value=None):
    ### remove the other kind of nuclei, only keep the tumor nuclei 
    mat = sio.loadmat(mat_file)
    mask = mat[var_name]
    if target_value is not None and var_name_extend1 is not None: 
        tumor = mat[var_name_extend1]
        ind_to_rm = np.where(tumor != target_value)[1] 
        for i in range(len(ind_to_rm)): 
            mask[mask==ind_to_rm[i]]=0
    return mask.astype(np.uint8)

from skimage.segmentation import clear_border
def save_one_mat_mask_rm_outerNuclei(mat_file, var_name, var_name_extend1=None, target_value=None, save_dir=None): 
    os.makedirs(save_dir, exist_ok=True)
    # mask = read_mat_mask(mat_file, var_name)
    mask = read_mat_mask_if_only_tumor(mat_file, var_name, var_name_extend1, target_value)
    mask = clear_border(mask)
    name = os.path.basename(mat_file).replace('.mat', '')
    im = Image.fromarray(mask)
    save_path = f'{save_dir}{os.sep}{name}.png'
    im.save(save_path)
    return save_path 

def save_mat_mask_rm_outerNuclei(src_dir, var_name, var_name_2, target_value, save_dir, n_workers):
    files = glob.glob(f'{src_dir}{os.sep}*.mat')
    size = len(files)
    with Pool(processes=n_workers) as pool:
        res = pool.starmap(
            save_one_mat_mask_rm_outerNuclei,
            zip(files, repeat(var_name, size), 
                repeat(var_name_2, size), repeat(target_value, size), 
                repeat(save_dir, size)))
    return res 


def extract_info_from_one_mat(mat_file, varnames, only_ndarray_size, target_value):
    varnames = [varnames] if isinstance(varnames, str) else varnames
    mat = sio.loadmat(mat_file)
    name = os.path.basename(mat_file).replace('.mat', '')
    res = {}
    res['name'] = name
    for varname in varnames:
        value = mat[varname]
        if isinstance(value, np.ndarray):
            if only_ndarray_size is not True:
                res[varname] = np.array2string(value, separator=',')
            if target_value is None:
                res[varname + f"_size"] = value.flatten().shape[0]
            else:
                res[varname+f"_{target_value}_size"] = np.sum(value.flatten() == target_value)
    return res


def extract_info_from_matfiles(src_dir, n_workers, varnames,
                               only_ndarray_size, target_value):
    """
    return pd.DataFrame
    """
    matfiles = glob.glob(f'{src_dir}{os.sep}*.mat')
    size = len(matfiles)
    with Pool(processes=n_workers) as pool:
        reses = pool.starmap(
            extract_info_from_one_mat,
            zip(matfiles, repeat(varnames, size),
                repeat(only_ndarray_size, size), repeat(target_value, size)))
    data = {}
    for res in reses:
        for k, v in res.items():
            if data.get(k) == None:
                data[k] = [v]
            else:
                data[k].append(v)
    df = pd.DataFrame(data=data)
    return df


def save_matfiles_info_to_df(src_dir,
                             n_workers,
                             varnames,
                             only_ndarray_size=True,
                             target_value = None,
                             save_path=None):
    df = extract_info_from_matfiles(src_dir, n_workers, varnames,
                                    only_ndarray_size, target_value)
    if save_path != None:
        save_results_to_pandas(df, save_path)
    return df
