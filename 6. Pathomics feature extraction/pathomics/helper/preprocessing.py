# get a low magnification image of a WSI
import os
import glob
import numpy as np
from PIL import Image
import multiprocessing
from multiprocessing import Pool, Manager
from itertools import repeat
from pathomics.image import WSI


def multiRunStarmap(f, *args):
    p = Pool(multiprocessing.cpu_count() // 3 * 2)
    n = 0
    args = list(args)
    for i in range(len(args)):
        if type(args[i]) == list:
            n = len(args[i])
            break

    for i in range(len(args)):
        if type(args[i]) != list:
            args[i] = repeat(args[i], n)
        elif len(args[i]) != n:
            args[i] = repeat(args[i], n)
    res = p.starmap(f, zip(*args))
    p.close()
    return res


def wsi2OnePatch(path, max_mag, curr_mag, coord, psize, save_dir, save_ext):
    x, y = coord
    scale = int(max_mag // curr_mag)
    wsi = WSI(path, {'scale': scale})
    roi = wsi.getRegion(x, y, psize, psize)
    im = Image.fromarray(roi)
    im.save(f'{save_dir}/{wsi.name}-{max_mag}-{curr_mag}-{x}-{y}{save_ext}')


def wsi2Patch(path, save_dir, save_ext, max_mag, curr_mag, psize):
    os.makedirs(save_dir, exist_ok=True)
    scale = int(max_mag // curr_mag)
    wsi = WSI(path, {'scale': scale})
    wsi.setIterator(psize)
    w, h = wsi.getSize()
    coords = wsi.genPatchCoords()
    #multiRunStarmapN(20, wsi2OnePatch, path, coords, psize, save_dir, save_ext)
    multiRunStarmap(wsi2OnePatch, path, max_mag, curr_mag, coords, psize,
                    save_dir, save_ext)


def mergeOnePatch(patch_path, save_scale, split, idx1, idx2):
    name = os.path.basename(patch_path)
    ext = '.' + name.split('.')[-1]
    name = name.replace(ext, '')
    im = Image.open(patch_path)
    w, h = im.size
    w = w // save_scale
    h = h // save_scale
    im = im.resize((w, h))
    im = np.array(im).astype(np.uint8)
    n = im.ndim
    splits = name.split(split)
    x = splits[idx1]
    y = splits[idx2]
    x = int(x) // save_scale
    y = int(y) // save_scale

    return x, y, im


def mergePatch(W, H, N, psize, patch_dir, patch_ext, save_scale, split, idx1,
               idx2):
    scale = save_scale
    patch_paths = glob.glob(f'{patch_dir}/*{patch_ext}')
    if N == 1:
        M = np.zeros((H // scale + psize, W // scale + psize))
    else:
        M = np.zeros((H // scale + psize, W // scale + psize, N))

    res = multiRunStarmap(mergeOnePatch, patch_paths, save_scale, split, idx1,
                          idx2)
    if N == 1:
        for re in res:
            x, y, im = re
            M[y:y + psize // scale, x:x + psize // scale] = im
        M = M[:H // scale, :W // scale]
    else:
        for re in res:
            x, y, im = re
            #print(H//scale,W//scale,y,x)
            M[y:y + psize // scale, x:x + psize // scale, :] = im
        M = M[:H // scale, :W // scale, :]
    M = M.astype(np.uint8)
    return M


def get_low_magnification_WSI(wsi_path, max_mag, curr_mag, save_path=None):
    psize = 256
    ext = '.' + wsi_path.split('.')[-1]
    name = os.path.basename(wsi_path).replace(ext, '')
    tmp_dir = '/tmp/adsfasdf{name}'
    wsi2Patch(wsi_path, tmp_dir, '.jpg', max_mag, curr_mag, psize)
    wsi = WSI(wsi_path, {'scale': int(max_mag / curr_mag)})
    W, H = wsi.getSize()
    np_im = mergePatch(W, H, 3, psize, tmp_dir, '.jpg', 1, '-', -2, -1)
    if save_path != None:
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        im = Image.fromarray(np_im)
        im.save(save_path)
    return np_im