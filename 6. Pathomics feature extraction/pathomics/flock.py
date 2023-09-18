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


class PathomicsFLocK(base.PathomicsFeaturesBase):

    def __init__(self, inputImage, inputMask, **kwargs):
        super(PathomicsFLocK, self).__init__(inputImage, inputMask, **kwargs)
        self.image = sitk.GetArrayViewFromImage(self.inputImage)
        self.mask = sitk.GetArrayViewFromImage(self.inputMask)
        if len(self.mask.shape) > 2:
            self.mask = self.mask[:, :, 0]
        self.bounds, self.image_intensity, self.feats = mask2bounds(
            self.mask, self.image, atts=['area'], **kwargs)
        self.features, _ = flock_feat_extractor(self.image_intensity,
                                                np.array(self.feats),
                                                ifShowFig=False,
                                                bandwidth=180)

    def getabs_area_minFeatureValue(self):
        return self.features[0]

    def getabs_area_maxFeatureValue(self):
        return self.features[1]

    def getabs_area_rangeFeatureValue(self):
        return self.features[2]

    def getabs_area_meanFeatureValue(self):
        return self.features[3]

    def getabs_area_medianFeatureValue(self):
        return self.features[4]

    def getabs_area_stdFeatureValue(self):
        return self.features[5]

    def getabs_area_kurtosisFeatureValue(self):
        return self.features[6]

    def getabs_area_skewnessFeatureValue(self):
        return self.features[7]

    def getcount_total_intersectFeatureValue(self):
        return self.features[8]

    def getcount_portion_intersectedFeatureValue(self):
        return self.features[9]

    def getmax_intersect_area_count_sum_10PercentFeatureValue(self):
        return self.features[10]

    def getmax_intersect_area_count_sum_20PercentFeatureValue(self):
        return self.features[11]

    def getmax_intersect_area_count_sum_30PercentFeatureValue(self):
        return self.features[12]

    def getmax_intersect_area_over_clust_sum_10PercentFeatureValue(self):
        return self.features[13]

    def getmax_intersect_area_over_clust_sum_20PercentFeatureValue(self):
        return self.features[14]

    def getmax_intersect_area_over_clust_sum_30PercentFeatureValue(self):
        return self.features[15]

    def getmin_intersect_area_stat_minFeatureValue(self):
        return self.features[16]

    def getmin_intersect_area_stat_maxFeatureValue(self):
        return self.features[17]

    def getmin_intersect_area_stat_rangeFeatureValue(self):
        return self.features[18]

    def getmin_intersect_area_stat_meanFeatureValue(self):
        return self.features[19]

    def getmin_intersect_area_stat_medianFeatureValue(self):
        return self.features[20]

    def getmin_intersect_area_stat_stdFeatureValue(self):
        return self.features[21]

    def getmin_intersect_area_stat_kurtosisFeatureValue(self):
        return self.features[22]

    def getmin_intersect_area_stat_skewnessFeatureValue(self):
        return self.features[23]

    def getvoronoi_area_stdFeatureValue(self):
        return self.features[24]

    def getvoronoi_area_meanFeatureValue(self):
        return self.features[25]

    def getvoronoi_area_min_maxFeatureValue(self):
        return self.features[26]

    def getvoronoi_area_disorderFeatureValue(self):
        return self.features[27]

    def getvoronoi_peri_stdFeatureValue(self):
        return self.features[28]

    def getvoronoi_peri_meanFeatureValue(self):
        return self.features[29]

    def getvoronoi_peri_min_maxFeatureValue(self):
        return self.features[30]

    def getvoronoi_peri_disorderFeatureValue(self):
        return self.features[31]

    def getvoronoi_chord_stdFeatureValue(self):
        return self.features[32]

    def getvoronoi_chord_meanFeatureValue(self):
        return self.features[33]

    def getvoronoi_chord_min_maxFeatureValue(self):
        return self.features[34]

    def getvoronoi_chord_disorderFeatureValue(self):
        return self.features[35]

    def getdelaunay_peri_min_maxFeatureValue(self):
        return self.features[36]

    def getdelaunay_peri_stdFeatureValue(self):
        return self.features[37]

    def getdelaunay_peri_meanFeatureValue(self):
        return self.features[38]

    def getdelaunay_peri_disorderFeatureValue(self):
        return self.features[39]

    def getdelaunay_area_min_maxFeatureValue(self):
        return self.features[40]

    def getdelaunay_area_stdFeatureValue(self):
        return self.features[41]

    def getdelaunay_area_meanFeatureValue(self):
        return self.features[42]

    def getdelaunay_area_disorderFeatureValue(self):
        return self.features[43]

    def getmst_edge_meanFeatureValue(self):
        return self.features[44]

    def getmst_edge_stdFeatureValue(self):
        return self.features[45]

    def getmst_edge_min_maxFeatureValue(self):
        return self.features[46]

    def getmst_edge_disorderFeatureValue(self):
        return self.features[47]

    def getnuc_sumFeatureValue(self):
        return self.features[48]

    def getnuc_vor_areaFeatureValue(self):
        return self.features[49]

    def getnuc_densityFeatureValue(self):
        return self.features[50]

    def getknn_std_3FeatureValue(self):
        return self.features[51]

    def getknn_mean_3FeatureValue(self):
        return self.features[52]

    def getknn_disorder_3FeatureValue(self):
        return self.features[53]

    def getknn_std_5FeatureValue(self):
        return self.features[54]

    def getknn_mean_5FeatureValue(self):
        return self.features[55]

    def getknn_disorder_5FeatureValue(self):
        return self.features[56]

    def getknn_std_7FeatureValue(self):
        return self.features[57]

    def getknn_mean_7FeatureValue(self):
        return self.features[58]

    def getknn_disorder_7FeatureValue(self):
        return self.features[59]

    def getrr_std_10FeatureValue(self):
        return self.features[60]

    def getrr_mean_10FeatureValue(self):
        return self.features[61]

    def getrr_disorder_10FeatureValue(self):
        return self.features[62]

    def getrr_std_20FeatureValue(self):
        return self.features[63]

    def getrr_mean_20FeatureValue(self):
        return self.features[64]

    def getrr_disorder_20FeatureValue(self):
        return self.features[65]

    def getrr_std_30FeatureValue(self):
        return self.features[66]

    def getrr_mean_30FeatureValue(self):
        return self.features[67]

    def getrr_disorder_30FeatureValue(self):
        return self.features[68]

    def getrr_std_40FeatureValue(self):
        return self.features[69]

    def getrr_mean_40FeatureValue(self):
        return self.features[70]

    def getrr_disorder_40FeatureValue(self):
        return self.features[71]

    def getrr_std_50FeatureValue(self):
        return self.features[72]

    def getrr_mean_50FeatureValue(self):
        return self.features[73]

    def getrr_disorder_50FeatureValue(self):
        return self.features[74]

    def getvoronoi_area_stdFeatureValue(self):
        return self.features[75]

    def getvoronoi_area_meanFeatureValue(self):
        return self.features[76]

    def getvoronoi_area_min_maxFeatureValue(self):
        return self.features[77]

    def getvoronoi_area_disorderFeatureValue(self):
        return self.features[78]

    def getvoronoi_peri_stdFeatureValue(self):
        return self.features[79]

    def getvoronoi_peri_meanFeatureValue(self):
        return self.features[80]

    def getvoronoi_peri_min_maxFeatureValue(self):
        return self.features[81]

    def getvoronoi_peri_disorderFeatureValue(self):
        return self.features[82]

    def getvoronoi_chord_stdFeatureValue(self):
        return self.features[83]

    def getvoronoi_chord_meanFeatureValue(self):
        return self.features[84]

    def getvoronoi_chord_min_maxFeatureValue(self):
        return self.features[85]

    def getvoronoi_chord_disorderFeatureValue(self):
        return self.features[86]

    def getdelaunay_peri_min_maxFeatureValue(self):
        return self.features[87]

    def getdelaunay_peri_stdFeatureValue(self):
        return self.features[88]

    def getdelaunay_peri_meanFeatureValue(self):
        return self.features[89]

    def getdelaunay_peri_disorderFeatureValue(self):
        return self.features[90]

    def getdelaunay_area_min_maxFeatureValue(self):
        return self.features[91]

    def getdelaunay_area_stdFeatureValue(self):
        return self.features[92]

    def getdelaunay_area_meanFeatureValue(self):
        return self.features[93]

    def getdelaunay_area_disorderFeatureValue(self):
        return self.features[94]

    def getmst_edge_meanFeatureValue(self):
        return self.features[95]

    def getmst_edge_stdFeatureValue(self):
        return self.features[96]

    def getmst_edge_min_maxFeatureValue(self):
        return self.features[97]

    def getmst_edge_disorderFeatureValue(self):
        return self.features[98]

    def getnuc_sumFeatureValue(self):
        return self.features[99]

    def getnuc_vor_areaFeatureValue(self):
        return self.features[100]

    def getnuc_densityFeatureValue(self):
        return self.features[101]

    def getknn_std_3FeatureValue(self):
        return self.features[102]

    def getknn_mean_3FeatureValue(self):
        return self.features[103]

    def getknn_disorder_3FeatureValue(self):
        return self.features[104]

    def getknn_std_5FeatureValue(self):
        return self.features[105]

    def getknn_mean_5FeatureValue(self):
        return self.features[106]

    def getknn_disorder_5FeatureValue(self):
        return self.features[107]

    def getknn_std_7FeatureValue(self):
        return self.features[108]

    def getknn_mean_7FeatureValue(self):
        return self.features[109]

    def getknn_disorder_7FeatureValue(self):
        return self.features[110]

    def getrr_std_10FeatureValue(self):
        return self.features[111]

    def getrr_mean_10FeatureValue(self):
        return self.features[112]

    def getrr_disorder_10FeatureValue(self):
        return self.features[113]

    def getrr_std_20FeatureValue(self):
        return self.features[114]

    def getrr_mean_20FeatureValue(self):
        return self.features[115]

    def getrr_disorder_20FeatureValue(self):
        return self.features[116]

    def getrr_std_30FeatureValue(self):
        return self.features[117]

    def getrr_mean_30FeatureValue(self):
        return self.features[118]

    def getrr_disorder_30FeatureValue(self):
        return self.features[119]

    def getrr_std_40FeatureValue(self):
        return self.features[120]

    def getrr_mean_40FeatureValue(self):
        return self.features[121]

    def getrr_disorder_40FeatureValue(self):
        return self.features[122]

    def getrr_std_50FeatureValue(self):
        return self.features[123]

    def getrr_mean_50FeatureValue(self):
        return self.features[124]

    def getrr_disorder_50FeatureValue(self):
        return self.features[125]

    def getflock_size_cluster_countFeatureValue(self):
        return self.features[126]

    def getflock_num_nuc_in_cluster_minFeatureValue(self):
        return self.features[127]

    def getflock_num_nuc_in_cluster_maxFeatureValue(self):
        return self.features[128]

    def getflock_num_nuc_in_cluster_rangeFeatureValue(self):
        return self.features[129]

    def getflock_num_nuc_in_cluster_meanFeatureValue(self):
        return self.features[130]

    def getflock_num_nuc_in_cluster_medianFeatureValue(self):
        return self.features[131]

    def getflock_num_nuc_in_cluster_stdFeatureValue(self):
        return self.features[132]

    def getflock_num_nuc_in_cluster_kurtosisFeatureValue(self):
        return self.features[133]

    def getflock_num_nuc_in_cluster_skewnessFeatureValue(self):
        return self.features[134]

    def getflock_nuc_density_over_size_minFeatureValue(self):
        return self.features[135]

    def getflock_nuc_density_over_size_maxFeatureValue(self):
        return self.features[136]

    def getflock_nuc_density_over_size_rangeFeatureValue(self):
        return self.features[137]

    def getflock_nuc_density_over_size_meanFeatureValue(self):
        return self.features[138]

    def getflock_nuc_density_over_size_medianFeatureValue(self):
        return self.features[139]

    def getflock_nuc_density_over_size_stdFeatureValue(self):
        return self.features[140]

    def getflock_nuc_density_over_size_kurtosisFeatureValue(self):
        return self.features[141]

    def getflock_nuc_density_over_size_skewnessFeatureValue(self):
        return self.features[142]

    def getflock_other_attr_minFeatureValue(self):
        return self.features[143]

    def getflock_other_attr_maxFeatureValue(self):
        return self.features[144]

    def getflock_other_attr_rangeFeatureValue(self):
        return self.features[145]

    def getflock_other_attr_meanFeatureValue(self):
        return self.features[146]

    def getflock_other_attr_medianFeatureValue(self):
        return self.features[147]

    def getflock_other_attr_stdFeatureValue(self):
        return self.features[148]

    def getflock_other_attr_kurtosisFeatureValue(self):
        return self.features[149]

    def getflock_other_attr_skewnessFeatureValue(self):
        return self.features[150]

    def getspan_var_dist_2_clust_cent_minFeatureValue(self):
        return self.features[151]

    def getspan_var_dist_2_clust_cent_maxFeatureValue(self):
        return self.features[152]

    def getspan_var_dist_2_clust_cent_rangeFeatureValue(self):
        return self.features[153]

    def getspan_var_dist_2_clust_cent_meanFeatureValue(self):
        return self.features[154]

    def getspan_var_dist_2_clust_cent_medianFeatureValue(self):
        return self.features[155]

    def getspan_var_dist_2_clust_cent_stdFeatureValue(self):
        return self.features[156]

    def getspan_var_dist_2_clust_cent_kurtosisFeatureValue(self):
        return self.features[157]

    def getspan_var_dist_2_clust_cent_skewnessFeatureValue(self):
        return self.features[158]

    def getpheno_enrichment_minFeatureValue(self):
        return self.features[159]

    def getpheno_enrichment_maxFeatureValue(self):
        return self.features[160]

    def getpheno_enrichment_rangeFeatureValue(self):
        return self.features[161]

    def getpheno_enrichment_meanFeatureValue(self):
        return self.features[162]

    def getpheno_enrichment_medianFeatureValue(self):
        return self.features[163]

    def getpheno_enrichment_stdFeatureValue(self):
        return self.features[164]

    def getpheno_enrichment_kurtosisFeatureValue(self):
        return self.features[165]

    def getpheno_enrichment_skewnessFeatureValue(self):
        return self.features[166]

    def getpheno_intra_same_type_stat_0_sumFeatureValue(self):
        return self.features[167]

    def getpheno_intra_same_type_stat_0_ratiosumFeatureValue(self):
        return self.features[168]

    def getpheno_inter_diff_type_stat_0_sumFeatureValue(self):
        return self.features[169]

    def getpheno_inter_diff_type_stat_0_ratiosumFeatureValue(self):
        return self.features[170]

    def getvoronoi_area_stdFeatureValue(self):
        return self.features[171]

    def getvoronoi_area_meanFeatureValue(self):
        return self.features[172]

    def getvoronoi_area_min_maxFeatureValue(self):
        return self.features[173]

    def getvoronoi_area_disorderFeatureValue(self):
        return self.features[174]

    def getvoronoi_peri_stdFeatureValue(self):
        return self.features[175]

    def getvoronoi_peri_meanFeatureValue(self):
        return self.features[176]

    def getvoronoi_peri_min_maxFeatureValue(self):
        return self.features[177]

    def getvoronoi_peri_disorderFeatureValue(self):
        return self.features[178]

    def getvoronoi_chord_stdFeatureValue(self):
        return self.features[179]

    def getvoronoi_chord_meanFeatureValue(self):
        return self.features[180]

    def getvoronoi_chord_min_maxFeatureValue(self):
        return self.features[181]

    def getvoronoi_chord_disorderFeatureValue(self):
        return self.features[182]

    def getdelaunay_peri_min_maxFeatureValue(self):
        return self.features[183]

    def getdelaunay_peri_stdFeatureValue(self):
        return self.features[184]

    def getdelaunay_peri_meanFeatureValue(self):
        return self.features[185]

    def getdelaunay_peri_disorderFeatureValue(self):
        return self.features[186]

    def getdelaunay_area_min_maxFeatureValue(self):
        return self.features[187]

    def getdelaunay_area_stdFeatureValue(self):
        return self.features[188]

    def getdelaunay_area_meanFeatureValue(self):
        return self.features[189]

    def getdelaunay_area_disorderFeatureValue(self):
        return self.features[190]

    def getmst_edge_meanFeatureValue(self):
        return self.features[191]

    def getmst_edge_stdFeatureValue(self):
        return self.features[192]

    def getmst_edge_min_maxFeatureValue(self):
        return self.features[193]

    def getmst_edge_disorderFeatureValue(self):
        return self.features[194]

    def getnuc_sumFeatureValue(self):
        return self.features[195]

    def getnuc_vor_areaFeatureValue(self):
        return self.features[196]

    def getnuc_densityFeatureValue(self):
        return self.features[197]

    def getknn_std_3FeatureValue(self):
        return self.features[198]

    def getknn_mean_3FeatureValue(self):
        return self.features[199]

    def getknn_disorder_3FeatureValue(self):
        return self.features[200]

    def getknn_std_5FeatureValue(self):
        return self.features[201]

    def getknn_mean_5FeatureValue(self):
        return self.features[202]

    def getknn_disorder_5FeatureValue(self):
        return self.features[203]

    def getknn_std_7FeatureValue(self):
        return self.features[204]

    def getknn_mean_7FeatureValue(self):
        return self.features[205]

    def getknn_disorder_7FeatureValue(self):
        return self.features[206]

    def getrr_std_10FeatureValue(self):
        return self.features[207]

    def getrr_mean_10FeatureValue(self):
        return self.features[208]

    def getrr_disorder_10FeatureValue(self):
        return self.features[209]

    def getrr_std_20FeatureValue(self):
        return self.features[210]

    def getrr_mean_20FeatureValue(self):
        return self.features[211]

    def getrr_disorder_20FeatureValue(self):
        return self.features[212]

    def getrr_std_30FeatureValue(self):
        return self.features[213]

    def getrr_mean_30FeatureValue(self):
        return self.features[214]

    def getrr_disorder_30FeatureValue(self):
        return self.features[215]

    def getrr_std_40FeatureValue(self):
        return self.features[216]

    def getrr_mean_40FeatureValue(self):
        return self.features[217]

    def getrr_disorder_40FeatureValue(self):
        return self.features[218]

    def getrr_std_50FeatureValue(self):
        return self.features[219]

    def getrr_mean_50FeatureValue(self):
        return self.features[220]

    def getrr_disorder_50FeatureValue(self):
        return self.features[221]
