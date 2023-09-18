from math import nan
import sys
from turtle import color  ##bj

#sys.path.append('./flock_extractor')

from pathomics.matlab.flock_extractor.flock.core import FlockCore, MeanShiftWrapper, FlockTypingKmeans
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
import numpy as np
from pathomics.matlab.flock_extractor.flock.feature import *
from .visualization import *
import networkx as nx


def showMeanShift(img, feat, cluster_info):
    fig, ax1 = plt.subplots(ncols=1, figsize=(9, 4))
    # plt.gray()
    label = cluster_info['label']
    member = cluster_info['member']
    member.keys()
    center_xy = cluster_info['center'][:, :2]
    # enum_dict = dict(zip(range(len(label)), label, feat[:,:2]))

    ax1.imshow(img)
    ax1.scatter(center_xy[:, 0], center_xy[:, 1], marker='*', color='r')
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] * 100
    for ind, m in enumerate(member.keys()):
        for i in member[m]:
            ax1.plot([center_xy[m, 0], feat[i, 0]],
                     [center_xy[m, 1], feat[i, 1]],
                     color=colors[ind])

    # plt.savefig('output_meanshift.png')
    plt.show()
    # plt.clf()


def flock_feat_extractor(img, feat_inp, ifShowFig=False, bandwidth=80):
    flock_core = FlockCore.build(feat_inp[:, :2], feat_inp[:, 2:], bandwidth)
    cluster_info = flock_core.cluster_init()

    if ifShowFig:
        ##show meanshift
        showMeanShift(img, feat_inp, cluster_info)

    # cluster_center_xy = cluster_info[FlockCore.KEY_CENTER][:, :2]
    labels = cluster_info[FlockCore.KEY_LABEL]
    nuc_centroids = feat_inp[:, :2]

    labels_unique = np.unique(labels)
    color_8bit = plt.get_cmap('rainbow', 2**5)

    flock_kmeans = FlockTypingKmeans.build(flock_core=flock_core, n_clusters=2)

    polygons = FlockPolygons(flock_core, flock_core.feat[:, :2],
                             flock_core.feat[:, 2:])

    p_dict = polygons.features

    flock_feat = FlockFeature(flock_core, flock_kmeans)
    feat = flock_feat.features

    # note -- the keys in flattened feature may contains duplicate as the corresponding numeric values are
    # the same type of features. The hierarchical interpretation of those keys which distinguishes them from each other
    # is lost after the flattening procedure.
    keys_list, feature_numerics_list = flock_feat.flattened_feature
    feature_vector = np.asarray(feature_numerics_list)

    flock_ms = FlockTypingMeanShift.build(flock_core)

    flock_feat_hdb = FlockFeature(flock_core, flock_ms)
    feat_hdb = flock_feat_hdb.features
    keys_hdb_list, feature_hdb_numerics_list = flock_feat_hdb.flattened_feature
    feature_hdb_numerics_list = [
        fe if not np.isnan(fe) else 0 for fe in feature_hdb_numerics_list
    ]

    if ifShowFig:
        # ##show meanshift
        # showMeanShift(img, feat_inp, cluster_info)

        ##show polygon
        plt.imshow(img)
        plot_flock_polygon(p_dict,
                           color_map=None,
                           lw=2,
                           linestyle='-',
                           max_count=100)
        # plt.show()
        area_dict = feat[FlockFeature.KEY_OUT_INTERSECT][
            FlockFeature.KEY_INTERMEDIATE][FlockFeature.KEY_AREA_DICT]
        spolygon_cache_dict = flock_feat.polygon_container.spolygon_cache_dict
        plot_intersect_from_area_dict(area_dict, spolygon_cache_dict)

        y_lim, x_lim = img.shape[:2]
        extent = 0, x_lim, 0, y_lim
        plt.imshow(img, extent=extent)
        coords = intersect_centroid(area_dict)
        mst, plot_coord = showcase_graph(coords)
        nodes = mst.nodes(data=True)

        pos = {key: (xx, yy) for key, (xx, yy) in enumerate(coords)}
        nx.draw(mst, pos=pos, node_size=500, node_color='orange')

        plot_flock_by_typing(flock_core, flock_typing=flock_kmeans)
        polygon_graph(flock_core, node_size=100, node_color='yellow')
        plt.imshow(img)
        plt.show()

    return feature_hdb_numerics_list, keys_hdb_list


#     print()
# from testing import load_dummy_data
# img, feat = load_dummy_data()
# feat_hdb = flock_feat(img, feat)
