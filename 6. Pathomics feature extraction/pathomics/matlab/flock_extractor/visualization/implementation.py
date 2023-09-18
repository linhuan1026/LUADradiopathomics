import matplotlib.pyplot as plt
import networkx
import numpy as np
from matplotlib.collections import LineCollection
from typing import Union, Set, Sequence, Callable, Tuple, List, Dict
from pathomics.matlab.flock_extractor.flock.core import FlockCore
from pathomics.matlab.flock_extractor.flock.feature import PolygonContainer
from pathomics.matlab.flock_extractor.flock.feature.util import GraphUtil
import networkx as nx
from networkx.algorithms.tree.mst import minimum_spanning_tree
from scipy.spatial import Delaunay, Voronoi

eps = np.finfo(dtype=np.float32).eps

color_8bit = plt.get_cmap('rainbow', 2**5)

KEY_MST = 'mst'
KEY_VORONOI = 'voronoi'
KEY_DELAUNAY = 'delaunay'


def plot_flock_helper(nuc_centroids: np.ndarray,
                      cluster_center_xy: np.ndarray,
                      labels: np.ndarray,
                      label_sets: Union[Set, Sequence],
                      color_map: Callable = color_8bit,
                      color_optional=None,
                      figsize: Tuple = (5, 5),
                      fig=None,
                      lw=0.5):
    """
    Helper function
    Args:
        nuc_centroids: XY coordinate of each nuclei centroids
        cluster_center_xy: XY coordinate of the center of each cluster.
        labels: Cluster label of each nuclei
        label_sets: A unique set of label indices
        color_map: Color Map to visualize the nuclei centroids by cluster id.
        color_optional: override colormap
        figsize: figsize signature for plt.figure

    Returns:

    """
    fig = plt.figure(figsize=figsize) if fig is None else fig
    for label_idx in label_sets:
        coords = nuc_centroids[labels == label_idx]
        if color_map is not None:
            point_color = np.tile(color_map(label_idx), (coords.shape[0], 1))
        else:
            point_color = color_optional
        plt.scatter(coords[:, 0], coords[:, 1], c=point_color, linewidths=1)
        center_curr = cluster_center_xy[label_idx]

        line_seg = LineCollection([[start, center_curr] for start in coords],
                                  lw=lw,
                                  colors=point_color)
        fig.axes[0].add_collection(line_seg)
    return fig


def plot_flock(
    flock_core: FlockCore,
    color_map: Callable = color_8bit,
    figsize: Tuple = (5, 5),
    fig=None,
    labels_unique=None,
    color_optional=None,
    lw=0.5,
):
    """
    Plot the flock clusters. Color of clusters can be defined by a color map.
    Connect each nuclei to center of the cluster with linesegs.
    Args:
        flock_core:
        color_map:
        figsize:
        fig:
        labels_unique:
        color_optional:
        lw:

    Returns:

    """
    cluster_info = flock_core.cluster_init()

    cluster_center_xy = cluster_info[FlockCore.KEY_CENTER][:, :2]
    labels = cluster_info[FlockCore.KEY_LABEL]
    nuc_centroids = flock_core.feat[:, :2]

    labels_unique = np.unique(
        labels) if labels_unique is None else labels_unique
    return plot_flock_helper(nuc_centroids,
                             cluster_center_xy,
                             labels,
                             labels_unique,
                             color_map,
                             color_optional=color_optional,
                             figsize=figsize,
                             fig=fig,
                             lw=lw)


def plot_flock_polygon(p_dict,
                       color_map: Union[Callable, None] = color_8bit,
                       lw=2,
                       linestyle='--',
                       max_count=np.inf,
                       fill=False,
                       **kwargs):
    """
    Plot the corresponding polygon of the flock.
    Args:
        p_dict:
        color_map: Color of the polygon contour line. if None --> black
        lw:
        linestyle:
        max_count:
        fill:
        **kwargs:

    Returns:

    """
    for count, poly_idx in enumerate(p_dict.keys()):
        if count >= max_count:
            break
        polygon_pts = p_dict[poly_idx]['polygon_contour']
        if polygon_pts.shape[0] <= 2:
            continue
        if color_map is not None:
            color = color_map(poly_idx)
        else:
            color = 'k'
        plt.plot(polygon_pts[:, 0],
                 polygon_pts[:, 1],
                 linestyle=linestyle,
                 lw=lw,
                 color=color,
                 **kwargs)
        st = np.stack([polygon_pts[-1], polygon_pts[0]])
        plt.plot(st[:, 0],
                 st[:, 1],
                 linestyle=linestyle,
                 lw=lw,
                 color=color,
                 **kwargs)
        if fill:
            plt.fill(polygon_pts[:, 0], polygon_pts[:, 1])


# refactor later
def plot_intersect_from_area_dict(area_dict, spolygon_cache_dict):
    """
    Area dict see area_dict_helper of FlockFeature. Plot intersection regions with colors.
    Args:
        area_dict:
        spolygon_cache_dict:

    Returns:

    """
    for src_idx, src_polygon in area_dict.items():
        for target_idx, target_polygon in src_polygon.items():
            area = target_polygon[PolygonContainer.KEY_INTER_AREA]
            if area <= eps:
                continue
            intersect_polygon = spolygon_cache_dict[src_idx].intersection(
                spolygon_cache_dict[target_idx])
            px, py = intersect_polygon.exterior.xy
            plt.fill(px, py)
    plt.show()


def _mst(xy_coord: np.ndarray) -> Tuple[networkx.Graph, np.ndarray]:
    """
    mst tree from coords
    Args:
        xy_coord:

    Returns:

    """
    dist_mat = GraphUtil.dist_mat(xy_coord)
    g = nx.convert_matrix.from_numpy_matrix(dist_mat)
    mst = minimum_spanning_tree(g)
    return mst, xy_coord


def _delaunay(xy_coords: np.ndarray) -> Tuple[networkx.Graph, np.ndarray]:
    """
    delaunay from coords
    Args:
        xy_coords:

    Returns:

    """
    delaunay_trig: Delaunay = Delaunay(xy_coords)
    simplices = delaunay_trig.simplices
    g = nx.Graph()
    for trig_path in simplices:
        nx.add_path(g, trig_path)
    return g, xy_coords


def _voronoi(xy_coords: np.ndarray) -> Tuple[networkx.Graph, np.ndarray]:
    """
    voronoi from coords
    Args:
        xy_coords:

    Returns:

    """
    vor: Voronoi = Voronoi(xy_coords)
    g = nx.Graph()
    rv: List[List[int]] = vor.ridge_vertices
    vertices = vor.vertices
    for pair_ind in rv:
        if any(np.asarray(pair_ind) < 0):
            continue
        nx.add_path(g, pair_ind)
    return g, vertices


def showcase_graph(
        xy_coord,
        graph_construct_name: str = 'mst'
) -> Tuple[networkx.Graph, np.ndarray]:
    """
    Construct the graph by given construction function, also returns the coordinates of nodes that
    can be used by the plot_graph function
    Args:
        xy_coord:
        graph_construct_name:

    Returns:

    """
    func_map: Dict[str, Callable[[
        np.ndarray,
    ], Tuple[networkx.Graph, np.ndarray]]] = {
        KEY_MST: _mst,
        KEY_VORONOI: _voronoi,
        KEY_DELAUNAY: _delaunay
    }
    graph_func = func_map[graph_construct_name]
    g, plot_coord = graph_func(xy_coord)
    return g, plot_coord


# refactor --> the searching of non-negative intersection area is redundant
def intersect_centroid(area_dict):
    """
    retrieve the centroid of intersection. helper function.
    Args:
        area_dict:

    Returns:

    """
    coord_list = []
    for src_idx, src_polygon in area_dict.items():
        for target_idx, target_polygon in src_polygon.items():
            area = target_polygon[PolygonContainer.KEY_INTER_AREA]
            if area <= eps:
                continue
            px, py = target_polygon[
                PolygonContainer.KEY_INTER_CENTROID].coords.xy
            row_coord = np.concatenate([px, py])
            coord_list.append(row_coord)
    return np.vstack(coord_list)


def plot_graph(coords,
               node_size,
               node_color,
               graph_construct_name: str = 'mst',
               **kwargs):
    """
    plot the showcase graph
    Args:
        coords:
        node_size:
        node_color:
        graph_construct_name:
        **kwargs:

    Returns:

    """
    g, plot_coord = showcase_graph(coords,
                                   graph_construct_name=graph_construct_name)
    pos = {key: (xx, yy) for key, (xx, yy) in enumerate(plot_coord)}
    obj = nx.draw(g,
                  pos=pos,
                  node_size=node_size,
                  node_color=node_color,
                  **kwargs)
    return obj


def intersection_graph(area_dict,
                       node_size=20,
                       node_color='red',
                       graph_construct_name: str = 'mst',
                       **kwargs):
    coords = intersect_centroid(area_dict)
    return plot_graph(coords,
                      node_size=node_size,
                      node_color=node_color,
                      graph_construct_name=graph_construct_name,
                      **kwargs)


def polygon_graph(flock_core, graph_construct_name: str = 'mst', **kwargs):
    polygon_center, _ = flock_core.split_feat(
        flock_core.clustering.clusters_centers_, flock_core.feat_slice_id)
    return plot_graph(polygon_center,
                      graph_construct_name=graph_construct_name,
                      **kwargs)


temp_dict = {0: 'aqua', 1: 'lime', 2: 'coral'}


def plot_flock_by_typing(flock_core,
                         flock_typing,
                         color_dict=temp_dict,
                         figsize=None):
    member_dict = flock_typing.cluster_info_[flock_typing.KEY_MEMBER]
    fig = plt.figure(figsize=figsize)
    for type_id, member_list in member_dict.items():
        color_typing = color_dict[type_id]
        plot_flock(flock_core,
                   color_map=None,
                   labels_unique=member_list,
                   color_optional=color_typing,
                   fig=fig)
    return fig
