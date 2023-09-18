"""
Flock Construction:
    1. x, y, prox feature. From Mask.
    Use X/Y coordinates. Top - left = (0, 0)
"""

import numpy as np
from abc import ABC
from typing import Dict, Sequence, Callable
from pathomics.matlab.flock_extractor.flock.core.clustering import AbstractClusterFuncWrapper, MeanShiftWrapper, KMeansWrapper


class ClusterSkeleton(ABC):
    """
    Base class of cluster (flock and phenotype)
    Note: contain all clusters, regardless of # of elements in each cluster.
    The minimum requirement of # of elements (e.g., 3 to form a polygon) should be defined in classes of higher levels
    """
    _cluster_flag: bool

    KEY_LABEL: str = 'label'
    KEY_CENTER: str = 'center'
    KEY_META: str = 'meta'
    KEY_ITER: str = 'n_iter'
    KEY_MEMBER: str = 'member'
    KEY_CLUSTER_LABEL_SET: str = 'unique_cluster'
    KEY_FEAT_SLICE: str = 'feat_slice_idx'

    @staticmethod
    def cluster2data_from_label(
            labels: np.ndarray) -> Dict[int, Sequence[int]]:
        """
        map from cluster id to members that belong to it.
        Args:
            labels: data -> cluster: indx = data point id. value = cluster label.

        Returns:
            Dict[int, Sequence[int]]: cluster -> data wherein the data is the list of idx of members
        """
        cl2data = dict()
        for point_idx, label_idx in enumerate(labels):
            cl2data[label_idx] = cl2data.get(label_idx, [])
            cl2data[label_idx].append(point_idx)

        return cl2data

    @staticmethod
    def cluster_helper(feature: np.ndarray,
                       clustering: AbstractClusterFuncWrapper,
                       feat_slice_id: int) -> Dict:
        """
        Helper function to execute the clustering algorithm and generating corresponding outputs.
        Args:
            feature: N * Dim. The input feature vector to cluster
            clustering: The given clustering algorithm, wrapped in AbstractClusterFuncWrapper
            feat_slice_id: the index in Dim that separates features into spatial coordinates and other attributes
        Returns:
            Dict: dict of cluster information: label of each input feature points, center of each cluster,
            member of each cluster, unique set of cluster labels, parameters of the clustering algorithm,
            and the feat_slice_id
        """
        cluster_labels: np.ndarray = clustering.fit_predict(feature, feature)
        cluster_centers: np.ndarray = clustering.clustering.cluster_centers_  # n_c * n_feat
        meta = clustering.get_params()
        cluster_members = ClusterSkeleton.cluster2data_from_label(
            cluster_labels)
        cluster_unique_set = np.unique(cluster_labels)
        out_dict = {
            ClusterSkeleton.KEY_LABEL: cluster_labels,
            ClusterSkeleton.KEY_CENTER: cluster_centers,
            ClusterSkeleton.KEY_MEMBER: cluster_members,
            ClusterSkeleton.KEY_CLUSTER_LABEL_SET: cluster_unique_set,
            ClusterSkeleton.KEY_META: meta,
            ClusterSkeleton.KEY_FEAT_SLICE: feat_slice_id,
        }

        return out_dict

    def cluster_init(self):
        """
        Perform clustering and return the cluster information. Set the __cluster_flag to be true as an indicator
        that the clustering model is already fit.
        Returns:
            Dict: dict of cluster information: label of each input feature points, center of each cluster,
            member of each cluster, unique set of cluster labels, parameters of the clustering algorithm,
            and the feat_slice_id
        """
        out_dict = ClusterSkeleton.cluster_helper(self.feat, self.clustering,
                                                  self.feat_slice_id)
        self.__cluster_flag = True
        self.cluster_info_.update(out_dict)
        return self.cluster_info_

    @staticmethod
    def merged_feat(spatial_feat: np.ndarray, extra_feat: np.ndarray):
        """
        Merge the spatial coordinates and other attributes into a single feature vector.
        Args:
            spatial_feat:
            extra_feat:

        Returns:

        """
        assert spatial_feat is not None
        if extra_feat is None:
            return spatial_feat
        assert spatial_feat.shape[0] == extra_feat.shape[0]
        # assert spatial_feat.ndim == extra_feat.ndim  ##chaned
        return np.hstack([spatial_feat, extra_feat])

    @staticmethod
    def split_feat(feat: np.ndarray, slice_id: int):
        """
        Reverse operation of merged_feat. Split the feat (N * Dim) by slice_id, such that feat[:, 0:slice_id] are
        returned as the spatial coordinates, and the rest are the other attributes.
        Args:
            feat:
            slice_id:

        Returns:

        """
        assert feat.ndim == 2
        spatial_feat = feat[:, :slice_id]
        extra_feat = feat[:, slice_id:]
        if extra_feat.size == 0:
            extra_feat = None
        return spatial_feat, extra_feat

    @property
    def spatial_feat(self):
        """
        Spatial coordinates
        Returns:

        """
        return self.__spatial_feat

    @property
    def extra_feat(self):
        """
        Other attributes such as the histologic proximities.
        Returns:

        """
        return self.__extra_feat

    @property
    def feat_slice_id(self) -> int:
        """
        Feature slice id that separates the spatial feature and other attributes
        Returns:

        """
        return self.__feat_slice_id

    def __init__(self, spatial_feat: np.ndarray, extra_feat: np.ndarray,
                 clustering: AbstractClusterFuncWrapper):
        """

        Args:
            spatial_feat: Target feature to cluster. Presumably the feature should contain coordinates
            extra_feat:
            clustering: The fitter for the clustering, e.g. sklearn's mean-shift cluster
        """
        super().__init__()
        self.__spatial_feat = spatial_feat
        self.__extra_feat = extra_feat
        self.__feat_slice_id = self.__spatial_feat.shape[-1]
        self.__feat = ClusterSkeleton.merged_feat(spatial_feat, extra_feat)
        self.__clustering: AbstractClusterFuncWrapper = clustering
        self.__cluster_info_ = dict()
        self.__cluster2data = dict()
        self.__cluster_flag = False
        self.cluster_init()

    @property
    def feat(self) -> np.ndarray:
        return self.__feat

    @property
    def clustering(self):
        return self.__clustering

    @property
    def cluster_info_(self):
        """
        Note: the outputs are defined in self.cluster_init
        Returns:

        """
        return self.__cluster_info_

    @property
    def cluster2data(self):
        return self.cluster_info_.get(FlockCore.KEY_MEMBER, None)


class FlockCore(ClusterSkeleton):
    """
    Basic Flock construction with feature and Meanshift clustering.
    """

    @classmethod
    def build(cls, spatial_feat, extra_feat, bandwidth=80):
        """
        Factory methods. Merge spatial feature and other attributes into a single feature vector and use mean-shift
        clustering to group the features.
        Args:
            spatial_feat: Coordinates
            extra_feat: Other attributes such as histologic proximities
            bandwidth: bandwidth parameter for mean-shift clustering

        Returns:

        """
        clustering = MeanShiftWrapper.build(bandwidth=bandwidth, seeds=None)
        obj = cls(spatial_feat=spatial_feat,
                  extra_feat=extra_feat,
                  clustering=clustering)
        return obj

    def __init__(self, spatial_feat: np.ndarray, extra_feat: np.ndarray,
                 clustering: MeanShiftWrapper):
        """

        Args:
            spatial_feat: Target feature to cluster. Presumably the feature should contain coordinates
            extra_feat:
            clustering: The fitter for the clustering, e.g. sklearn's mean-shift cluster
        """

        super().__init__(spatial_feat, extra_feat, clustering)


class FlockTyping(ClusterSkeleton):
    """
    Base class for flock phenotyping. Difference to the FlockCore is this class operates on clusters rather than
    individual feature points

    """

    def __init__(self, flock_core: FlockCore,
                 re_cluster: AbstractClusterFuncWrapper):
        feat = flock_core.cluster_info_[ClusterSkeleton.KEY_CENTER]
        split_idx = flock_core.feat_slice_id
        spatial_feat, extra_feat = ClusterSkeleton.split_feat(feat, split_idx)
        super().__init__(spatial_feat, extra_feat, re_cluster)
        self.__core = flock_core
        # cluster again

    @property
    def core(self):
        """
        The FlockCore object it operates on.
        Returns:

        """

        return self.__core

    @classmethod
    def build(cls, flock_core: FlockCore, construct: Callable, *args,
              **kwargs):
        cluster_func = construct(*args, **kwargs)
        return cls(flock_core, cluster_func)


class FlockTypingKmeans(FlockTyping):
    """
    Grouping flock phenotypes by kmeans
    """

    def __init__(self, flock_core: FlockCore, kmeans: KMeansWrapper):
        super().__init__(flock_core, kmeans)

    @classmethod
    def build(cls, flock_core: FlockCore, *args, **kwargs):
        construct_func = KMeansWrapper.build
        return super().build(flock_core, construct_func, *args, **kwargs)


class FlockTypingMeanShift(FlockTyping):
    """
    Grouping flock phenotypes by Mean-shift: it does not generate fixed number of output --> different number
    of phenotypes per tile input --> different feature vector size
    """

    def __init__(self, flock_core: FlockCore, mean_shift: MeanShiftWrapper):
        super().__init__(flock_core, mean_shift)

    @classmethod
    def build(cls, flock_core: FlockCore, *args, **kwargs):
        construct_func = MeanShiftWrapper.build
        return super().build(flock_core, construct_func, *args, **kwargs)


# class FlockTypingKHDB(FlockTyping):
#
#     def __init__(self, flock_core: FlockCore, kmeans: KMeansWrapper):
#         super().__init__(flock_core, kmeans)
#
#     @classmethod
#     def build(cls, flock_core: FlockCore, *args, **kwargs):
#         construct_func = HDBScanWrapper.build
#         return super().build(flock_core, construct_func, *args, **kwargs)
