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


class PathomicsProps(base.PathomicsFeaturesBase):

    def __init__(self, inputImage, inputMask, **kwargs):
        super(PathomicsProps, self).__init__(inputImage, inputMask,
                                                **kwargs)
        self.image = sitk.GetArrayViewFromImage(self.inputImage)
        self.mask = sitk.GetArrayViewFromImage(self.inputMask)
        if len(self.mask.shape) > 2:
            self.mask = self.mask[:, :, 0]
        self.bounds, self.image_intensity, self.feats = mask2bounds(
            self.mask, self.image, atts=['area'], **kwargs)
        self.features, _ = extract_nuclei_props(
                            fname_mask=self.mask, fname_intensity=self.image_intensity)

    def getcentroid_xFeatureValue(self):
        return self.features[:,0]

    def getcentroid_yFeatureValue(self):
        return self.features[:,1]

    def getareaFeatureValue(self):
        return self.features[:,2]

    def getaxis_major_lengthFeatureValue(self):
        return self.features[:,3]

    def getaxis_minor_lengthFeatureValue(self):
        return self.features[:,4]

    def geteccentricityFeatureValue(self):
        return self.features[:,5]

    def getequivalent_diameter_areaFeatureValue(self):
        return self.features[:,6]

    def geteuler_numberFeatureValue(self):
        return self.features[:,7]

    def getextentFeatureValue(self):
        return self.features[:,8]

    def getferet_diameter_maxFeatureValue(self):
        return self.features[:,9]

    def getorientationFeatureValue(self):
        return self.features[:,10]

    def getperimeterFeatureValue(self):
        return self.features[:,11]

    def getperimeter_croftonFeatureValue(self):
        return self.features[:,12]

    def getsolidityFeatureValue(self):
        return self.features[:,13]

    def getintensity_maxFeatureValue(self):
        return self.features[:,14]

    def getintensity_meanFeatureValue(self):
        return self.features[:,15]

    def getintensity_minFeatureValue(self):
        return self.features[:,16]

    def getintensity_stdFeatureValue(self):
        return self.features[:,17]
