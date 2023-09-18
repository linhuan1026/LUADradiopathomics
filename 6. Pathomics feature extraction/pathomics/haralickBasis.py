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


class PathomicsHaralickBasis(base.PathomicsFeaturesBase):

    def __init__(self, inputImage, inputMask, **kwargs):
        super(PathomicsHaralickBasis, self).__init__(inputImage, inputMask,
                                                **kwargs)
        self.image = sitk.GetArrayViewFromImage(self.inputImage)
        self.mask = sitk.GetArrayViewFromImage(self.inputMask)
        if len(self.mask.shape) > 2:
            self.mask = self.mask[:, :, 0]
        self.bounds, self.image_intensity, self.feats = mask2bounds(
            self.mask, self.image, atts=['area'], **kwargs)
        self.features, _ = Lharalick_img_nuclei_wise_basis(
            fname_mask=self.mask, fname_intensity=self.image_intensity)

    def getcontrast_energyFeatureValue(self):
        return self.features[:,0]

    def getcontrast_inverse_momentFeatureValue(self):
        return self.features[:,1]

    def getcontrast_aveFeatureValue(self):
        return self.features[:,2]

    def getcontrast_varFeatureValue(self):
        return self.features[:,3]

    def getcontrast_entropyFeatureValue(self):
        return self.features[:,4]

    def getintensity_aveFeatureValue(self):
        return self.features[:,5]

    def getintensity_varianceFeatureValue(self):
        return self.features[:,6]

    def getintensity_entropyFeatureValue(self):
        return self.features[:,7]

    def getentropyFeatureValue(self):
        return self.features[:,8]

    def getenergyFeatureValue(self):
        return self.features[:,9]

    def getcorrelationFeatureValue(self):
        return self.features[:,10]

    def getinformation_measure1FeatureValue(self):
        return self.features[:,11]

    def getinformation_measure2FeatureValue(self):
        return self.features[:,12]
