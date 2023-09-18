import numpy
from six.moves import range
import scipy.io as sio
import SimpleITK as sitk
from PIL import Image
import numpy as np
import skimage
from skimage import measure
from skimage import morphology
import pandas as pd
from pathomics import base
from .matlab import *


class PathomicsCGT(base.PathomicsFeaturesBase):

    def __init__(self, inputImage, inputMask, **kwargs):
        super(PathomicsCGT, self).__init__(inputImage, inputMask, **kwargs)
        self.image = sitk.GetArrayViewFromImage(self.inputImage)
        self.mask = sitk.GetArrayViewFromImage(self.inputMask)
        if len(self.mask.shape) > 2:
            self.mask = self.mask[:, :, 0]
        self.bounds, self.image_intensity, self.feats = mask2bounds(
            self.mask, self.image, atts=['area'], **kwargs)
        CGTfeats, c_matrix, info, feats_, network, edges = extract_CGT_feats(
            self.bounds)
        self.features = CGTfeats

    def getmean_tensor_contrast_energyFeatureValue(self):
        return self.features[0]

    def getstandard_deviation_tensor_contrast_energyFeatureValue(self):
        return self.features[1]

    def getrange_tensor_contrast_energyFeatureValue(self):
        return self.features[2]

    def getmean_tensor_contrast_inverse_momentFeatureValue(self):
        return self.features[3]

    def getstandard_deviation_tensor_contrast_inverse_momentFeatureValue(self):
        return self.features[4]

    def getrange_tensor_contrast_inverse_momentFeatureValue(self):
        return self.features[5]

    def getmean_tensor_contrast_aveFeatureValue(self):
        return self.features[6]

    def getstandard_deviation_tensor_contrast_aveFeatureValue(self):
        return self.features[7]

    def getrange_tensor_contrast_aveFeatureValue(self):
        return self.features[8]

    def getmean_tensor_contrast_varFeatureValue(self):
        return self.features[9]

    def getstandard_deviation_tensor_contrast_varFeatureValue(self):
        return self.features[10]

    def getrange_tensor_contrast_varFeatureValue(self):
        return self.features[11]

    def getmean_tensor_contrast_entropyFeatureValue(self):
        return self.features[12]

    def getstandard_deviation_tensor_contrast_entropyFeatureValue(self):
        return self.features[13]

    def getrange_tensor_contrast_entropyFeatureValue(self):
        return self.features[14]

    def getmean_tensor_intensity_aveFeatureValue(self):
        return self.features[15]

    def getstandard_deviation_tensor_intensity_aveFeatureValue(self):
        return self.features[16]

    def getrange_tensor_intensity_aveFeatureValue(self):
        return self.features[17]

    def getmean_tensor_intensity_varianceFeatureValue(self):
        return self.features[18]

    def getstandard_deviation_tensor_intensity_varianceFeatureValue(self):
        return self.features[19]

    def getrange_tensor_intensity_varianceFeatureValue(self):
        return self.features[20]

    def getmean_tensor_intensity_entropyFeatureValue(self):
        return self.features[21]

    def getstandard_deviation_tensor_intensity_entropyFeatureValue(self):
        return self.features[22]

    def getrange_tensor_intensity_entropyFeatureValue(self):
        return self.features[23]

    def getmean_tensor_entropyFeatureValue(self):
        return self.features[24]

    def getstandard_deviation_tensor_entropyFeatureValue(self):
        return self.features[25]

    def getrange_tensor_entropyFeatureValue(self):
        return self.features[26]

    def getmean_tensor_energyFeatureValue(self):
        return self.features[27]

    def getstandard_deviation_tensor_energyFeatureValue(self):
        return self.features[28]

    def getrange_tensor_energyFeatureValue(self):
        return self.features[29]

    def getmean_tensor_correlationFeatureValue(self):
        return self.features[30]

    def getstandard_deviation_tensor_correlationFeatureValue(self):
        return self.features[31]

    def getrange_tensor_correlationFeatureValue(self):
        return self.features[32]

    def getmean_tensor_information_measure1FeatureValue(self):
        return self.features[33]

    def getstandard_deviation_tensor_information_measure1FeatureValue(self):
        return self.features[34]

    def getrange_tensor_information_measure1FeatureValue(self):
        return self.features[35]

    def getmean_tensor_information_measure2FeatureValue(self):
        return self.features[36]

    def getstandard_deviation_tensor_information_measure2FeatureValue(self):
        return self.features[37]

    def getrange_tensor_information_measure2FeatureValue(self):
        return self.features[38]
