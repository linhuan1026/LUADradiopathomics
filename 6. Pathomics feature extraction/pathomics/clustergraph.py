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


class PathomicsClusterGraph(base.PathomicsFeaturesBase):

    def __init__(self, inputImage, inputMask, **kwargs):
        super(PathomicsClusterGraph, self).__init__(inputImage, inputMask,
                                                    **kwargs)
        self.image = sitk.GetArrayViewFromImage(self.inputImage)
        self.mask = sitk.GetArrayViewFromImage(self.inputMask)
        if len(self.mask.shape) > 2:
            self.mask = self.mask[:, :, 0]
        self.bounds, self.image_intensity, self.feats = mask2bounds(
            self.mask, self.image, atts=['area'], **kwargs)
        self.features, _ = extract_cluster_graph_feats(self.bounds)

    def getNumber_of_NodesFeatureValue(self):
        return self.features[0]

    def getNumber_of_EdgesFeatureValue(self):
        return self.features[1]

    def getAverage_DegreeFeatureValue(self):
        return self.features[2]

    def getAverage_EccentricityFeatureValue(self):
        return self.features[3]

    def getDiameterFeatureValue(self):
        return self.features[4]

    def getRadiusFeatureValue(self):
        return self.features[5]

    def getAverage_Eccentricity_90percentFeatureValue(self):
        return self.features[6]

    def getDiameter_90percentFeatureValue(self):
        return self.features[7]

    def getRadius_90percentFeatureValue(self):
        return self.features[8]

    def getAverage_Path_LengthFeatureValue(self):
        return self.features[9]

    def getClustering_Coefficient_CFeatureValue(self):
        return self.features[10]

    def getClustering_Coefficient_DFeatureValue(self):
        return self.features[11]

    def getClustering_Coefficient_EFeatureValue(self):
        return self.features[12]

    def getNumber_of_connected_componentsFeatureValue(self):
        return self.features[13]

    def getgiant_connected_component_ratioFeatureValue(self):
        return self.features[14]

    def getaverage_connected_component_sizeFeatureValue(self):
        return self.features[15]

    def getnumber_isolated_nodesFeatureValue(self):
        return self.features[16]

    def getpercentage_isolated_nodesFeatureValue(self):
        return self.features[17]

    def getnumber_end_nodesFeatureValue(self):
        return self.features[18]

    def getpercentage_end_nodesFeatureValue(self):
        return self.features[19]

    def getnumber_central_nodesFeatureValue(self):
        return self.features[20]

    def getpercentage_central_nodesFeatureValue(self):
        return self.features[21]

    def getmean_edge_lengthFeatureValue(self):
        return self.features[22]

    def getstandard_deviation_edge_lengthFeatureValue(self):
        return self.features[23]

    def getskewness_edge_lengthFeatureValue(self):
        return self.features[24]

    def getkurtosis_edge_lengthFeatureValue(self):
        return self.features[25]
