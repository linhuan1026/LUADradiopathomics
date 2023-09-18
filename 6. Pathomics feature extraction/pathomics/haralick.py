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


class PathomicsHaralick(base.PathomicsFeaturesBase):

    def __init__(self, inputImage, inputMask, **kwargs):
        super(PathomicsHaralick, self).__init__(inputImage, inputMask,
                                                **kwargs)
        self.image = sitk.GetArrayViewFromImage(self.inputImage)
        self.mask = sitk.GetArrayViewFromImage(self.inputMask)
        if len(self.mask.shape) > 2:
            self.mask = self.mask[:, :, 0]
        self.bounds, self.image_intensity, self.feats = mask2bounds(
            self.mask, self.image, atts=['area'], **kwargs)
        self.features, _ = Lharalick_img_nuclei_wise(
            fname_mask=self.mask, fname_intensity=self.image_intensity)

    def getmean_contrast_energyFeatureValue(self):
        return self.features[0]

    def getmean_contrast_inverse_momentFeatureValue(self):
        return self.features[1]

    def getmean_contrast_aveFeatureValue(self):
        return self.features[2]

    def getmean_contrast_varFeatureValue(self):
        return self.features[3]

    def getmean_contrast_entropyFeatureValue(self):
        return self.features[4]

    def getmean_intensity_aveFeatureValue(self):
        return self.features[5]

    def getmean_intensity_varianceFeatureValue(self):
        return self.features[6]

    def getmean_intensity_entropyFeatureValue(self):
        return self.features[7]

    def getmean_entropyFeatureValue(self):
        return self.features[8]

    def getmean_energyFeatureValue(self):
        return self.features[9]

    def getmean_correlationFeatureValue(self):
        return self.features[10]

    def getmean_information_measure1FeatureValue(self):
        return self.features[11]

    def getmean_information_measure2FeatureValue(self):
        return self.features[12]

    def getmedian_contrast_energyFeatureValue(self):
        return self.features[13]

    def getmedian_contrast_inverse_momentFeatureValue(self):
        return self.features[14]

    def getmedian_contrast_aveFeatureValue(self):
        return self.features[15]

    def getmedian_contrast_varFeatureValue(self):
        return self.features[16]

    def getmedian_contrast_entropyFeatureValue(self):
        return self.features[17]

    def getmedian_intensity_aveFeatureValue(self):
        return self.features[18]

    def getmedian_intensity_varianceFeatureValue(self):
        return self.features[19]

    def getmedian_intensity_entropyFeatureValue(self):
        return self.features[20]

    def getmedian_entropyFeatureValue(self):
        return self.features[21]

    def getmedian_energyFeatureValue(self):
        return self.features[22]

    def getmedian_correlationFeatureValue(self):
        return self.features[23]

    def getmedian_information_measure1FeatureValue(self):
        return self.features[24]

    def getmedian_information_measure2FeatureValue(self):
        return self.features[25]

    def getstd_contrast_energyFeatureValue(self):
        return self.features[26]

    def getstd_contrast_inverse_momentFeatureValue(self):
        return self.features[27]

    def getstd_contrast_aveFeatureValue(self):
        return self.features[28]

    def getstd_contrast_varFeatureValue(self):
        return self.features[29]

    def getstd_contrast_entropyFeatureValue(self):
        return self.features[30]

    def getstd_intensity_aveFeatureValue(self):
        return self.features[31]

    def getstd_intensity_varianceFeatureValue(self):
        return self.features[32]

    def getstd_intensity_entropyFeatureValue(self):
        return self.features[33]

    def getstd_entropyFeatureValue(self):
        return self.features[34]

    def getstd_energyFeatureValue(self):
        return self.features[35]

    def getstd_correlationFeatureValue(self):
        return self.features[36]

    def getstd_information_measure1FeatureValue(self):
        return self.features[37]

    def getstd_information_measure2FeatureValue(self):
        return self.features[38]

    def getrange_contrast_energyFeatureValue(self):
        return self.features[39]

    def getrange_contrast_inverse_momentFeatureValue(self):
        return self.features[40]

    def getrange_contrast_aveFeatureValue(self):
        return self.features[41]

    def getrange_contrast_varFeatureValue(self):
        return self.features[42]

    def getrange_contrast_entropyFeatureValue(self):
        return self.features[43]

    def getrange_intensity_aveFeatureValue(self):
        return self.features[44]

    def getrange_intensity_varianceFeatureValue(self):
        return self.features[45]

    def getrange_intensity_entropyFeatureValue(self):
        return self.features[46]

    def getrange_entropyFeatureValue(self):
        return self.features[47]

    def getrange_energyFeatureValue(self):
        return self.features[48]

    def getrange_correlationFeatureValue(self):
        return self.features[49]

    def getrange_information_measure1FeatureValue(self):
        return self.features[50]

    def getrange_information_measure2FeatureValue(self):
        return self.features[51]

    def getkurtosis_contrast_energyFeatureValue(self):
        return self.features[52]

    def getkurtosis_contrast_inverse_momentFeatureValue(self):
        return self.features[53]

    def getkurtosis_contrast_aveFeatureValue(self):
        return self.features[54]

    def getkurtosis_contrast_varFeatureValue(self):
        return self.features[55]

    def getkurtosis_contrast_entropyFeatureValue(self):
        return self.features[56]

    def getkurtosis_intensity_aveFeatureValue(self):
        return self.features[57]

    def getkurtosis_intensity_varianceFeatureValue(self):
        return self.features[58]

    def getkurtosis_intensity_entropyFeatureValue(self):
        return self.features[59]

    def getkurtosis_entropyFeatureValue(self):
        return self.features[60]

    def getkurtosis_energyFeatureValue(self):
        return self.features[61]

    def getkurtosis_correlationFeatureValue(self):
        return self.features[62]

    def getkurtosis_information_measure1FeatureValue(self):
        return self.features[63]

    def getkurtosis_information_measure2FeatureValue(self):
        return self.features[64]

    def getskewness_contrast_energyFeatureValue(self):
        return self.features[65]

    def getskewness_contrast_inverse_momentFeatureValue(self):
        return self.features[66]

    def getskewness_contrast_aveFeatureValue(self):
        return self.features[67]

    def getskewness_contrast_varFeatureValue(self):
        return self.features[68]

    def getskewness_contrast_entropyFeatureValue(self):
        return self.features[69]

    def getskewness_intensity_aveFeatureValue(self):
        return self.features[70]

    def getskewness_intensity_varianceFeatureValue(self):
        return self.features[71]

    def getskewness_intensity_entropyFeatureValue(self):
        return self.features[72]

    def getskewness_entropyFeatureValue(self):
        return self.features[73]

    def getskewness_energyFeatureValue(self):
        return self.features[74]

    def getskewness_correlationFeatureValue(self):
        return self.features[75]

    def getskewness_information_measure1FeatureValue(self):
        return self.features[76]

    def getskewness_information_measure2FeatureValue(self):
        return self.features[77]
