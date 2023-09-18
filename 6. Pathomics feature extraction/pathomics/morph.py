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
import os


class PathomicsMorph(base.PathomicsFeaturesBase):

    def __init__(self, inputImage, inputMask, **kwargs):
        super(PathomicsMorph, self).__init__(inputImage, inputMask, **kwargs)
        self.image = sitk.GetArrayViewFromImage(self.inputImage)
        self.mask = sitk.GetArrayViewFromImage(self.inputMask)
        if len(self.mask.shape) > 2:
            self.mask = self.mask[:, :, 0]
        self.bounds, self.image_intensity, self.feats = mask2bounds(
            self.mask, self.image, atts=['area'], **kwargs)
        self.features = extract_morph_feats(self.bounds)

    def getMean_Area_RatioFeatureValue(self):
        return self.features[0]

    def getMean_Distance_RatioFeatureValue(self):
        return self.features[1]

    def getMean_Standard_Deviation_of_DistanceFeatureValue(self):
        return self.features[2]

    def getMean_Variance_of_DistanceFeatureValue(self):
        return self.features[3]

    def getMean_LongorShort_Distance_RatioFeatureValue(self):
        return self.features[4]

    def getMean_Perimeter_RatioFeatureValue(self):
        return self.features[5]

    def getMean_SmoothnessFeatureValue(self):
        return self.features[6]

    def getMean_Invariant_Moment_1FeatureValue(self):
        return self.features[7]

    def getMean_Invariant_Moment_2FeatureValue(self):
        return self.features[8]

    def getMean_Invariant_Moment_3FeatureValue(self):
        return self.features[9]

    def getMean_Invariant_Moment_4FeatureValue(self):
        return self.features[10]

    def getMean_Invariant_Moment_5FeatureValue(self):
        return self.features[11]

    def getMean_Invariant_Moment_6FeatureValue(self):
        return self.features[12]

    def getMean_Invariant_Moment_7FeatureValue(self):
        return self.features[13]

    def getMean_Fractal_DimensionFeatureValue(self):
        return self.features[14]

    def getMean_Fourier_Descriptor_1FeatureValue(self):
        return self.features[15]

    def getMean_Fourier_Descriptor_2FeatureValue(self):
        return self.features[16]

    def getMean_Fourier_Descriptor_3FeatureValue(self):
        return self.features[17]

    def getMean_Fourier_Descriptor_4FeatureValue(self):
        return self.features[18]

    def getMean_Fourier_Descriptor_5FeatureValue(self):
        return self.features[19]

    def getMean_Fourier_Descriptor_6FeatureValue(self):
        return self.features[20]

    def getMean_Fourier_Descriptor_7FeatureValue(self):
        return self.features[21]

    def getMean_Fourier_Descriptor_8FeatureValue(self):
        return self.features[22]

    def getMean_Fourier_Descriptor_9FeatureValue(self):
        return self.features[23]

    def getMean_Fourier_Descriptor_10FeatureValue(self):
        return self.features[24]

    def getStandard_Deviation_Area_RatioFeatureValue(self):
        return self.features[25]

    def getStandard_Deviation_Distance_RatioFeatureValue(self):
        return self.features[26]

    def getStandard_Deviation_Standard_Deviation_of_DistanceFeatureValue(self):
        return self.features[27]

    def getStandard_Deviation_Variance_of_DistanceFeatureValue(self):
        return self.features[28]

    def getStandard_Deviation_LongorShort_Distance_RatioFeatureValue(self):
        return self.features[29]

    def getStandard_Deviation_Perimeter_RatioFeatureValue(self):
        return self.features[30]

    def getStandard_Deviation_SmoothnessFeatureValue(self):
        return self.features[31]

    def getStandard_Deviation_Invariant_Moment_1FeatureValue(self):
        return self.features[32]

    def getStandard_Deviation_Invariant_Moment_2FeatureValue(self):
        return self.features[33]

    def getStandard_Deviation_Invariant_Moment_3FeatureValue(self):
        return self.features[34]

    def getStandard_Deviation_Invariant_Moment_4FeatureValue(self):
        return self.features[35]

    def getStandard_Deviation_Invariant_Moment_5FeatureValue(self):
        return self.features[36]

    def getStandard_Deviation_Invariant_Moment_6FeatureValue(self):
        return self.features[37]

    def getStandard_Deviation_Invariant_Moment_7FeatureValue(self):
        return self.features[38]

    def getStandard_Deviation_Fractal_DimensionFeatureValue(self):
        return self.features[39]

    def getStandard_Deviation_Fourier_Descriptor_1FeatureValue(self):
        return self.features[40]

    def getStandard_Deviation_Fourier_Descriptor_2FeatureValue(self):
        return self.features[41]

    def getStandard_Deviation_Fourier_Descriptor_3FeatureValue(self):
        return self.features[42]

    def getStandard_Deviation_Fourier_Descriptor_4FeatureValue(self):
        return self.features[43]

    def getStandard_Deviation_Fourier_Descriptor_5FeatureValue(self):
        return self.features[44]

    def getStandard_Deviation_Fourier_Descriptor_6FeatureValue(self):
        return self.features[45]

    def getStandard_Deviation_Fourier_Descriptor_7FeatureValue(self):
        return self.features[46]

    def getStandard_Deviation_Fourier_Descriptor_8FeatureValue(self):
        return self.features[47]

    def getStandard_Deviation_Fourier_Descriptor_9FeatureValue(self):
        return self.features[48]

    def getStandard_Deviation_Fourier_Descriptor_10FeatureValue(self):
        return self.features[49]

    def getMedian_Area_RatioFeatureValue(self):
        return self.features[50]

    def getMedian_Distance_RatioFeatureValue(self):
        return self.features[51]

    def getMedian_Standard_Deviation_of_DistanceFeatureValue(self):
        return self.features[52]

    def getMedian_Variance_of_DistanceFeatureValue(self):
        return self.features[53]

    def getMedian_LongorShort_Distance_RatioFeatureValue(self):
        return self.features[54]

    def getMedian_Perimeter_RatioFeatureValue(self):
        return self.features[55]

    def getMedian_SmoothnessFeatureValue(self):
        return self.features[56]

    def getMedian_Invariant_Moment_1FeatureValue(self):
        return self.features[57]

    def getMedian_Invariant_Moment_2FeatureValue(self):
        return self.features[58]

    def getMedian_Invariant_Moment_3FeatureValue(self):
        return self.features[59]

    def getMedian_Invariant_Moment_4FeatureValue(self):
        return self.features[60]

    def getMedian_Invariant_Moment_5FeatureValue(self):
        return self.features[61]

    def getMedian_Invariant_Moment_6FeatureValue(self):
        return self.features[62]

    def getMedian_Invariant_Moment_7FeatureValue(self):
        return self.features[63]

    def getMedian_Fractal_DimensionFeatureValue(self):
        return self.features[64]

    def getMedian_Fourier_Descriptor_1FeatureValue(self):
        return self.features[65]

    def getMedian_Fourier_Descriptor_2FeatureValue(self):
        return self.features[66]

    def getMedian_Fourier_Descriptor_3FeatureValue(self):
        return self.features[67]

    def getMedian_Fourier_Descriptor_4FeatureValue(self):
        return self.features[68]

    def getMedian_Fourier_Descriptor_5FeatureValue(self):
        return self.features[69]

    def getMedian_Fourier_Descriptor_6FeatureValue(self):
        return self.features[70]

    def getMedian_Fourier_Descriptor_7FeatureValue(self):
        return self.features[71]

    def getMedian_Fourier_Descriptor_8FeatureValue(self):
        return self.features[72]

    def getMedian_Fourier_Descriptor_9FeatureValue(self):
        return self.features[73]

    def getMedian_Fourier_Descriptor_10FeatureValue(self):
        return self.features[74]

    def getMin_or_Max_Area_RatioFeatureValue(self):
        return self.features[75]

    def getMin_or_Max_Distance_RatioFeatureValue(self):
        return self.features[76]

    def getMin_or_Max_Standard_Deviation_of_DistanceFeatureValue(self):
        return self.features[77]

    def getMin_or_Max_Variance_of_DistanceFeatureValue(self):
        return self.features[78]

    def getMin_or_Max_LongorShort_Distance_RatioFeatureValue(self):
        return self.features[79]

    def getMin_or_Max_Perimeter_RatioFeatureValue(self):
        return self.features[80]

    def getMin_or_Max_SmoothnessFeatureValue(self):
        return self.features[81]

    def getMin_or_Max_Invariant_Moment_1FeatureValue(self):
        return self.features[82]

    def getMin_or_Max_Invariant_Moment_2FeatureValue(self):
        return self.features[83]

    def getMin_or_Max_Invariant_Moment_3FeatureValue(self):
        return self.features[84]

    def getMin_or_Max_Invariant_Moment_4FeatureValue(self):
        return self.features[85]

    def getMin_or_Max_Invariant_Moment_5FeatureValue(self):
        return self.features[86]

    def getMin_or_Max_Invariant_Moment_6FeatureValue(self):
        return self.features[87]

    def getMin_or_Max_Invariant_Moment_7FeatureValue(self):
        return self.features[88]

    def getMin_or_Max_Fractal_DimensionFeatureValue(self):
        return self.features[89]

    def getMin_or_Max_Fourier_Descriptor_1FeatureValue(self):
        return self.features[90]

    def getMin_or_Max_Fourier_Descriptor_2FeatureValue(self):
        return self.features[91]

    def getMin_or_Max_Fourier_Descriptor_3FeatureValue(self):
        return self.features[92]

    def getMin_or_Max_Fourier_Descriptor_4FeatureValue(self):
        return self.features[93]

    def getMin_or_Max_Fourier_Descriptor_5FeatureValue(self):
        return self.features[94]

    def getMin_or_Max_Fourier_Descriptor_6FeatureValue(self):
        return self.features[95]

    def getMin_or_Max_Fourier_Descriptor_7FeatureValue(self):
        return self.features[96]

    def getMin_or_Max_Fourier_Descriptor_8FeatureValue(self):
        return self.features[97]

    def getMin_or_Max_Fourier_Descriptor_9FeatureValue(self):
        return self.features[98]

    def getMin_or_Max_Fourier_Descriptor_10FeatureValue(self):
        return self.features[99]
