from pathomics.matlab.flock_extractor.flock.core import *
from pathomics.matlab.flock_extractor.flock.feature.implementation.base import FeatureSkeleton
from scipy.spatial import ConvexHull
from lazy_property import LazyProperty
from typing import Dict


class FlockPolygons(FeatureSkeleton):
    """
    Obtain 'PolygonFeatures' for each polygon from flocks.
    Dict: Cluster ID to feature
    """

    @classmethod
    def build(cls, flock_cluster: FlockCore, minimum_size_inclusive: int = 5):
        return cls(flock_cluster,
                   flock_cluster.spatial_feat,
                   flock_cluster.extra_feat,
                   minimum_size_inclusive=minimum_size_inclusive)

    def __init__(self,
                 flock_cluster: ClusterSkeleton,
                 centroids: np.ndarray,
                 extra_feat: np.ndarray,
                 minimum_size_inclusive: int = 5):
        """

        Args:
            flock_cluster:
            centroids: Centroids of each member (e.g. nuclei). N_member * 2
            extra_feat: e.g. mean intensity. N_member * D
            minimum_size_inclusive: thresholding how many points required for construction. Inclusive (>=)
        """
        self.__flock_cluster = flock_cluster
        self.__centroids = centroids
        self.__extra_feat = extra_feat
        self.__minimum_size_inclusive = minimum_size_inclusive

    @property
    def minimum_size_inclusive(self):
        return self.__minimum_size_inclusive

    @staticmethod
    def polygon_info(flock_core: ClusterSkeleton,
                     centroids_all: np.ndarray,
                     extra_feat_all: np.ndarray,
                     minimum_size_inclusive: int = 3):
        """
        Describe each Flock cluster as a polygon for further area/intersection calculation.
        Calculate the relevant stats for downstreaming processings.
        Args:
            flock_core:
            centroids_all: centroid coord of all elements (e.g. nuclei)
            extra_feat_all: other attributes of all elements
            minimum_size_inclusive: if # of member < minimum_size_inclusive --> skip this cluster

        Returns:
            Dict: cluster_id given by flock_core --> Relevant PolygonFeatures
        """
        # extra_feat = flock_core.feat[:, 2:]
        # centroids = flock_core.feat[:, :2]
        cluster_polygon_dict: Dict[int, Dict] = dict()
        # traverse each cluster id
        for cluster_id in flock_core.cluster_info_[
                FlockCore.KEY_CLUSTER_LABEL_SET]:
            # fetch the idx of all members
            curr_member_idx = flock_core.cluster_info_[
                FlockCore.KEY_MEMBER][cluster_id]
            # find the centroid coordinates of each member by the idx
            if len(curr_member_idx) < minimum_size_inclusive:
                continue
            cluster_data_centroids = centroids_all[curr_member_idx]
            # extra feat
            cluster_attr = extra_feat_all[curr_member_idx]
            poly_feat = PolygonFeature(
                cluster_data_centroids,
                cluster_attr,
                cluster_info=flock_core.cluster_info_).features()
            cluster_polygon_dict[cluster_id] = poly_feat
        return cluster_polygon_dict

    @LazyProperty
    def features(self) -> Dict:
        """

        Returns:
            Dict <cluster_id given by flock_core> -> polygon features
        """
        return FlockPolygons.polygon_info(self.__flock_cluster,
                                          self.__centroids, self.__extra_feat,
                                          self.__minimum_size_inclusive)


class PolygonFeature(FeatureSkeleton):
    """
    Features obtained from a single polygon, derived from a flock cluster.
    Features of individual polygons. For now this class consider the case of cluster with 1 or 2 vertices for
    future development.
    For convenience, structures of higher levels (e.g. class FlockPolygons) will barricade
    cases of cluster of 1-2 vertices.
    Features include: area/std of distance of spatial coords to center/
    std of distance of other attributes to center/number of nuclei

    Intermediate: contour line /cluster info
    """
    KEY_POLYGON_AREA: str = 'polygon_area'
    KEY_CONTOUR: str = 'polygon_contour'
    KEY_MEAN_COORD: str = 'mean_centroid'
    KEY_DIST_STD: str = 'dist_span_std'
    KEY_ATTR_STD: str = 'other_attr'
    KEY_NUM_NUC: str = 'num_nuc_in_clust'
    KEY_CLUSTER_MISC: str = 'cluster_info'

    @staticmethod
    def euclidean_vectorized(arr1: np.ndarray, arr2: np.ndarray):
        """

        Args:
            arr1: N * 2
            arr2: N * 2 or 1 * 2

        Returns:

        """
        return np.sqrt(np.square(arr1 - arr2).sum(axis=1))

    @staticmethod
    def to_center_std(in_vec: np.ndarray):
        """
        std([distances])
        Args:
            in_vec: N * D

        Returns:

        """
        mean_vec = in_vec.mean(axis=0)
        std_value = PolygonFeature.euclidean_vectorized(in_vec, mean_vec).std()
        return std_value

    @staticmethod
    def __output_helper(area, contour, dist_to_center_std, attr_to_center_std,
                        num_nuc, cluster_info):
        """
        Write feature into the dict
        Args:
            area:
            contour:
            dist_to_center_std:
            attr_to_center_std:
            num_nuc:
            cluster_info:

        Returns:

        """
        result = dict()
        result[PolygonFeature.KEY_POLYGON_AREA] = area
        result[PolygonFeature.KEY_CONTOUR] = contour
        result[PolygonFeature.KEY_DIST_STD] = dist_to_center_std
        result[PolygonFeature.KEY_ATTR_STD] = attr_to_center_std
        result[PolygonFeature.KEY_NUM_NUC] = num_nuc
        result[PolygonFeature.KEY_CLUSTER_MISC] = cluster_info
        return result

    @staticmethod
    def cluster_polygon_helper(cluster_data_centroids,
                               cluster_attr,
                               cluster_info=None):
        """
        If there are at least three vertices (nuclei)
        calculate the desired features.
        Args:
            cluster_data_centroids:
            cluster_attr:
            cluster_info:

        Returns:

        """
        assert cluster_data_centroids.shape[0] >= 3
        qhull = ConvexHull(cluster_data_centroids)
        contour = cluster_data_centroids[qhull.vertices]
        area = qhull.area

        dist_to_center_std = PolygonFeature.to_center_std(
            cluster_data_centroids)
        # mean_coord = cluster_data_centroids.mean(axis=0)
        # dist_to_center_std = PolygonFeature.euclidean_vectorized(cluster_data_centroids, mean_coord).std()

        attr_to_center_std = PolygonFeature.to_center_std(cluster_attr)
        num_nuc = cluster_data_centroids.shape[0]
        result = PolygonFeature.__output_helper(area,
                                                contour,
                                                dist_to_center_std,
                                                attr_to_center_std,
                                                num_nuc,
                                                cluster_info=cluster_info)
        return result

    @staticmethod
    def cluster_non_polygon(cluster_data_centroids,
                            cluster_attr,
                            cluster_info=None,
                            nuc_boundary=None):
        """
        If only two vertices are observed --> set relevant numeric features that require the polygon to 0.
        That includes the area and distances to the center.
        Args:
            cluster_data_centroids:
            cluster_attr:
            cluster_info:
            nuc_boundary:

        Returns:

        """
        # todo cluster_attr edge case
        num_nuc = cluster_data_centroids.shape[0]
        assert num_nuc <= 2
        # may need to change cluster_data_centroids to boundary of nuclei

        result = PolygonFeature.__output_helper(0,
                                                cluster_data_centroids,
                                                0.,
                                                0.,
                                                num_nuc,
                                                cluster_info=cluster_info)
        return result

    @staticmethod
    def cluster_polygon(cluster_data_centroids,
                        cluster_attr,
                        cluster_info=None,
                        nuc_boundary=None):
        """
        Calculate features. Predefined values are given for cluster of size 2 or 1.
        Args:
            cluster_data_centroids:
            cluster_attr:
            cluster_info:
            nuc_boundary:

        Returns:

        """
        if cluster_data_centroids.shape[0] <= 2:
            return PolygonFeature.cluster_non_polygon(
                cluster_data_centroids,
                cluster_attr,
                cluster_info=cluster_info,
                nuc_boundary=nuc_boundary)
        return PolygonFeature.cluster_polygon_helper(cluster_data_centroids,
                                                     cluster_attr,
                                                     cluster_info=cluster_info)

    def __init__(self,
                 cluster_data_centroids,
                 cluster_attr,
                 cluster_info=None,
                 nuc_boundary=None):
        super().__init__()
        self.__cluster_data_centroids = cluster_data_centroids
        self.__cluster_attr = cluster_attr
        self.cluster_info = cluster_info
        self.__nuc_boundary = nuc_boundary

    def features(self) -> Dict:
        return PolygonFeature.cluster_polygon(self.__cluster_data_centroids,
                                              self.__cluster_attr,
                                              cluster_info=self.cluster_info,
                                              nuc_boundary=self.__nuc_boundary)
