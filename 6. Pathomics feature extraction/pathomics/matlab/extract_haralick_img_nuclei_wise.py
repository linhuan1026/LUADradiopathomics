from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import measure
from skimage import feature
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import warnings
import scipy
from .extract_haralick_img_nuclei_wise_basis import Lharalick_img_nuclei_wise_basis

def Lharalick_img_nuclei_wise(fname_mask='test_dense.png',
                              fname_intensity='test_intensity_image.png'):


    feats, feats_names = Lharalick_img_nuclei_wise_basis(fname_mask, fname_intensity)

    #### %% statistics: mean, median, standard deviation, range, kurtosis, skewness , across bounds for each haralick feature
    #### % check if the current nuclear morpholgy has no haralick features
    modifier = ['mean', 'median', 'std', 'range', 'kurtosis', 'skewness']
    mean = np.mean(feats, 0)
    median = np.median(feats, 0)
    std = np.std(feats, 0)
    range = np.max(feats, 0) - np.min(feats, 0)
    kurtosis = scipy.stats.kurtosis(
        feats, axis=0, fisher=False
    )  ##% kurtosis  ##default using Fisher's definition.. intend to use Pearson’s definition of kurtosis. MATLAB
    skewness = scipy.stats.skew(feats, 0)

    ## statistics
    featvals = np.hstack([mean, median, std, range, kurtosis, skewness])
    feats_names_total = [
        'HaralickNuWise:' + m + '_' + f + '' for m in modifier
        for f in feats_names
    ]

    return featvals.tolist(), feats_names_total


# fname_mask = 'test_dense.png'
# fname_intensity = 'test_intensity_image.png'

# featvals, feats_names = Lharalick_img_nuclei_wise(fname_mask=fname_mask, fname_intensity=fname_intensity)
# print(featvals)
# print()
