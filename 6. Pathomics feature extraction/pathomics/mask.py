import numpy as np
import SimpleITK as sitk
from skimage import io, color, img_as_ubyte, morphology


def getHistoQCMask(img, **kwargs):
    if isinstance(img, sitk.SimpleITK.Image):
        img = sitk.GetArrayViewFromImage(img)
    if isinstance(img, np.ndarray) is not True:
        raise ValueError('getHistoQCMask must have a numpy array')
    upper_thresh = .9
    lower_var = 10
    lower_thresh = -float('inf')
    upper_var = float('inf')
    img_var = img.std(axis=2)
    map_var = np.bitwise_and(img_var > lower_var, img_var < upper_var)
    img = color.rgb2gray(img)
    map = np.bitwise_and(img > lower_thresh, img < upper_thresh)
    map = np.bitwise_and(map, map_var)
    map = (map > 0)
    map = map.astype(np.uint8)
    return map