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


class PathomicsGraph(base.PathomicsFeaturesBase):

    def __init__(self, inputImage, inputMask, **kwargs):
        super(PathomicsGraph, self).__init__(inputImage, inputMask, **kwargs)
        self.image = sitk.GetArrayViewFromImage(self.inputImage)
        self.mask = sitk.GetArrayViewFromImage(self.inputMask)
        if len(self.mask.shape) > 2:
            self.mask = self.mask[:, :, 0]
        self.bounds, self.image_intensity, self.feats = mask2bounds(
            self.mask, self.image, atts=['area'], **kwargs)
        self.features = get_graph_features(self.bounds)

    def getArea_Standard_DeviationFeatureValue(self):
        return self.features[0]

    def getArea_AverageFeatureValue(self):
        return self.features[1]

    def getArea_Minimum_or_MaximumFeatureValue(self):
        return self.features[2]

    def getArea_DisorderFeatureValue(self):
        return self.features[3]

    def getPerimeter_Standard_DeviationFeatureValue(self):
        return self.features[4]

    def getPerimeter_AverageFeatureValue(self):
        return self.features[5]

    def getPerimeter_Minimum_or_MaximumFeatureValue(self):
        return self.features[6]

    def getPerimeter_DisorderFeatureValue(self):
        return self.features[7]

    def getChord_Standard_DeviationFeatureValue(self):
        return self.features[8]

    def getChord_AverageFeatureValue(self):
        return self.features[9]

    def getChord_Minimum_or_MaximumFeatureValue(self):
        return self.features[10]

    def getChord_DisorderFeatureValue(self):
        return self.features[11]

    def getSide_Length_Minimum_or_MaximumFeatureValue(self):
        return self.features[12]

    def getSide_Length_Standard_DeviationFeatureValue(self):
        return self.features[13]

    def getSide_Length_AverageFeatureValue(self):
        return self.features[14]

    def getSide_Length_DisorderFeatureValue(self):
        return self.features[15]

    def getTriangle_Area_Minimum_or_MaximumFeatureValue(self):
        return self.features[16]

    def getTriangle_Area_Standard_DeviationFeatureValue(self):
        return self.features[17]

    def getTriangle_Area_AverageFeatureValue(self):
        return self.features[18]

    def getTriangle_Area_DisorderFeatureValue(self):
        return self.features[19]

    def getMST_Edge_Length_AverageFeatureValue(self):
        return self.features[20]

    def getMST_Edge_Length_Standard_DeviationFeatureValue(self):
        return self.features[21]

    def getMST_Edge_Length_Minimum_or_MaximumFeatureValue(self):
        return self.features[22]

    def getMST_Edge_Length_DisorderFeatureValue(self):
        return self.features[23]

    def getArea_of_polygonsFeatureValue(self):
        return self.features[24]

    def getNumber_of_PolygonsFeatureValue(self):
        return self.features[25]

    def getDensity_of_PolygonsFeatureValue(self):
        return self.features[26]

    def getAverage_distance_to_3_Nearest_NeighborsFeatureValue(self):
        return self.features[27]

    def getAverage_distance_to_5_Nearest_NeighborsFeatureValue(self):
        return self.features[28]

    def getAverage_distance_to_7_Nearest_NeighborsFeatureValue(self):
        return self.features[29]

    def getStandard_Deviation_distance_to_3_Nearest_NeighborsFeatureValue(
            self):
        return self.features[30]

    def getStandard_Deviation_distance_to_5_Nearest_NeighborsFeatureValue(
            self):
        return self.features[31]

    def getStandard_Deviation_distance_to_7_Nearest_NeighborsFeatureValue(
            self):
        return self.features[32]

    def getDisorder_of_distance_to_3_Nearest_NeighborsFeatureValue(self):
        return self.features[33]

    def getDisorder_of_distance_to_5_Nearest_NeighborsFeatureValue(self):
        return self.features[34]

    def getDisorder_of_distance_to_7_Nearest_NeighborsFeatureValue(self):
        return self.features[35]

    def getAvg_Nearest_Neighbors_in_a_10_Pixel_RadiusFeatureValue(self):
        return self.features[36]

    def getAvg_Nearest_Neighbors_in_a_20_Pixel_RadiusFeatureValue(self):
        return self.features[37]

    def getAvg_Nearest_Neighbors_in_a_30_Pixel_RadiusFeatureValue(self):
        return self.features[38]

    def getAvg_Nearest_Neighbors_in_a_40_Pixel_RadiusFeatureValue(self):
        return self.features[39]

    def getAvg_Nearest_Neighbors_in_a_50_Pixel_RadiusFeatureValue(self):
        return self.features[40]

    def getStandard_Deviation_Nearest_Neighbors_in_a_10_Pixel_RadiusFeatureValue(
            self):
        return self.features[41]

    def getStandard_Deviation_Nearest_Neighbors_in_a_20_Pixel_RadiusFeatureValue(
            self):
        return self.features[42]

    def getStandard_Deviation_Nearest_Neighbors_in_a_30_Pixel_RadiusFeatureValue(
            self):
        return self.features[43]

    def getStandard_Deviation_Nearest_Neighbors_in_a_40_Pixel_RadiusFeatureValue(
            self):
        return self.features[44]

    def getStandard_Deviation_Nearest_Neighbors_in_a_50_Pixel_RadiusFeatureValue(
            self):
        return self.features[45]

    def getDisorder_of_Nearest_Neighbors_in_a_10_Pixel_RadiusFeatureValue(
            self):
        return self.features[46]

    def getDisorder_of_Nearest_Neighbors_in_a_20_Pixel_RadiusFeatureValue(
            self):
        return self.features[47]

    def getDisorder_of_Nearest_Neighbors_in_a_30_Pixel_RadiusFeatureValue(
            self):
        return self.features[48]

    def getDisorder_of_Nearest_Neighbors_in_a_40_Pixel_RadiusFeatureValue(
            self):
        return self.features[49]

    def getDisorder_of_Nearest_Neighbors_in_a_50_Pixel_RadiusFeatureValue(
            self):
        return self.features[50]
