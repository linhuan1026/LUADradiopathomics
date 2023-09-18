from flock.core import FlockCore, MeanShiftWrapper, FlockTypingKmeans
from testing import load_dummy_data
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
import numpy as np
from flock.feature import *
from visualization import *
import networkx as nx


img, feat = load_dummy_data()
flock_core = FlockCore.build(feat[:, :2], feat[:, 2:], 80)
cluster_info = flock_core.cluster_init()

cluster_center_xy = cluster_info[FlockCore.KEY_CENTER][:, :2]
labels = cluster_info[FlockCore.KEY_LABEL]
nuc_centroids = feat[:, :2]

labels_unique = np.unique(labels)
color_8bit = plt.get_cmap('rainbow', 2**5)

# fig = plt.figure(figsize=(5, 5))
# plt.imshow(img)
# for label_idx in labels_unique:
#     coords = nuc_centroids[labels == label_idx]
#     point_color = np.tile(color_8bit(label_idx), (coords.shape[0], 1))
#     plt.scatter(coords[:, 0], coords[:, 1], c=point_color, linewidths=0.5)
#     center_curr = cluster_center_xy[label_idx]
#
#     line_seg = LineCollection([[start, center_curr] for start in coords], colors=point_color)
#     fig.axes[0].add_collection(line_seg)
# plt.show()

flock_kmeans = FlockTypingKmeans.build(flock_core=flock_core, n_clusters=2)

polygons = FlockPolygons(flock_core, flock_core.feat[:, :2], flock_core.feat[:, 2:])

p_dict = polygons.features

# for poly_idx in p_dict.keys():
#     polygon_pts = p_dict[poly_idx]['polygon_contour']
#     if polygon_pts.shape[0] <= 2:
#         continue
#
#     plt.plot(polygon_pts[:, 0], polygon_pts[:, 1],
#              'k--', lw=2)
#     st = np.stack([polygon_pts[-1], polygon_pts[0]])
#     plt.plot(st[:, 0], st[:, 1],
#              'k--', lw=2)
#     plt.show()
#
# for poly_idx in p_dict.keys():
#     pts1 = p_dict[1]['polygon_contour']
#     pts2 = p_dict[poly_idx]['polygon_contour']
#     p1 = SPolygon(pts1)
#     if pts2.shape[0] <= 2:
#         continue
#     p2 = SPolygon(pts2)
#     print(p1.intersection(p2).area)

flock_feat = FlockFeature(flock_core, flock_kmeans)
feat = flock_feat.features

# note -- the keys in flattened feature may contains duplicate as the corresponding numeric values are
# the same type of features. The hierarchical interpretation of those keys which distinguishes them from each other
# is lost after the flattening procedure.
keys_list, feature_numerics_list = flock_feat.flattened_feature
feature_vector = np.asarray(feature_numerics_list)

flock_ms= FlockTypingMeanShift.build(flock_core)

flock_feat_hdb = FlockFeature(flock_core, flock_ms)
feat_hdb = flock_feat_hdb.features

plt.imshow(img)
plot_flock_polygon(p_dict, color_map=None, lw=2, linestyle='-', max_count=20)
# plt.show()
area_dict = feat[FlockFeature.KEY_OUT_INTERSECT][FlockFeature.KEY_INTERMEDIATE][FlockFeature.KEY_AREA_DICT]
spolygon_cache_dict = flock_feat.polygon_container.spolygon_cache_dict
plot_intersect_from_area_dict(area_dict, spolygon_cache_dict)
plt.show()

y_lim, x_lim = img.shape[:-1]
extent = 0, x_lim, 0, y_lim
plt.imshow(img, extent=extent)
coords = intersect_centroid(area_dict)
mst, plot_coord = showcase_graph(coords)
nodes = mst.nodes(data=True)

pos = {key: (xx, yy) for key, (xx, yy) in enumerate(coords)}
nx.draw(mst,
        pos=pos,
        node_size=500,
        node_color='orange')

plot_flock_by_typing(flock_core, flock_typing=flock_kmeans)
polygon_graph(flock_core, node_size=100, node_color='yellow')
plt.imshow(img)
# print()