"""
Low level graph features.
"""
from pathomics.matlab.flock_extractor.flock.feature.implementation.base import NamedFeatureSkeleton
from pathomics.matlab.flock_extractor.flock.feature.util import GraphUtil, FeatureStore

from pathomics.matlab.flock_extractor.flock.feature.implementation.graph_level.meta_data import _FeatName
from scipy.spatial import Delaunay, Voronoi
from scipy.sparse.csgraph import minimum_spanning_tree

import numpy as np
from abc import abstractmethod
from typing import Any, Sequence, Dict, List, Union
from lazy_property import LazyProperty
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

eps = np.finfo(float).eps


def nan_arr():
    return np.asarray([
        np.nan,
    ])


class GraphFeature(NamedFeatureSkeleton):
    """
    Base class for all graph features.
    Features are not reduced (i.e. by first order statistics like mean/std) but remain as a list of values (e.g.,
    a list of area values of all simplices in the delaunay graph)
    """
    KEY_GRAPH_OBJ: str = 'graph_obj'
    # __feature_store: FeatureStore
    __xy_coord_all: np.ndarray

    @staticmethod
    def validate_float_list(in_list: Union[List[float], List[np.ndarray]],
                            fill: Union[float, np.ndarray] = np.nan):
        """
        Check if the list is empty. If empty, return a list with default values (by default, nan)
        Args:
            in_list:
            fill:

        Returns:

        """
        if len(in_list) == 0:
            logger.debug('Empty')
            return [fill]
        return in_list

    @property
    def xy_coord_all(self) -> np.ndarray:
        """
        All node coordinates for the graph.
        Returns:

        """
        return self.__xy_coord_all

    @property
    @abstractmethod
    def graph_obj(self) -> Any:
        """
        Simply expose whatever was constructed.
        Returns:

        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def is_coord_valid(xy_coord):
        """
        Interface to define the edge case (invalid)
        Args:
            xy_coord:

        Returns:

        """
        raise NotImplementedError

    @property
    def valid_flag(self):
        """
        Whether it is edge case (invalid) or not
        Returns:

        """
        return self.__valid_flag

    def __init__(self, xy_coord: np.ndarray, init_value=nan_arr()):
        super().__init__(init_value)
        self.__xy_coord_all = np.atleast_2d(xy_coord)
        self.__valid_flag = self.is_coord_valid(xy_coord)

    @abstractmethod
    def feature_helper_valid(self,
                             feature_store: FeatureStore) -> FeatureStore:
        """
        Helper function to generate features and save it in FeatureStore object.
        Args:
            feature_store:

        Returns:

        """
        ...

    @LazyProperty
    def features(self) -> Dict:
        """
        If edge case --> return the feature store object with default values.
        Otherwise, calculate and write the feature values into the FeatureStore
        Returns:

        """
        if not self.__valid_flag:
            return self.feature_store.feature_dict
        return self.feature_helper_valid(self.feature_store).feature_dict

    def feature_names(self):
        ...

    @staticmethod
    def by_flag_helper(valid: bool, valid_case, invalid_case):
        """
        A helper function to select values by conditions. If valid then return valid_case.
        Args:
            valid:
            valid_case:
            invalid_case:

        Returns:

        """
        if valid:
            return valid_case
        return invalid_case

    def by_flag(self, valid_case, invalid_case):
        """
        Todo - not used yet.
        Args:
            valid_case:
            invalid_case:

        Returns:

        """
        return GraphFeature.by_flag_helper(self.valid_flag, valid_case,
                                           invalid_case)


class VoronoiFeature(GraphFeature):
    """
    Voronoi feature that calculates area/perimeter/chord length of each voronoi polygon.
    """
    KEY_CHORD: str = 'chord'
    KEY_PERI: str = 'perimeter'
    KEY_AREA: str = 'area'

    __voronoi: Union[Voronoi, None]
    __region_points: Union[np.ndarray, None]
    __regions_all: Union[Sequence[Sequence[int]], None]
    __region_qualified_dict: Union[Dict[int, Sequence[int]], None]

    def init_value_curate(self):
        self._feature_set_value(VoronoiFeature.KEY_CHORD, [np.nan])

    @staticmethod
    def is_coord_valid(xy_coord: np.ndarray):
        assert xy_coord.ndim == 2
        return xy_coord.shape[0] >= 4

    @property
    def feature_names(self):
        return [
            VoronoiFeature.KEY_CHORD, VoronoiFeature.KEY_PERI,
            VoronoiFeature.KEY_AREA, VoronoiFeature.KEY_GRAPH_OBJ
        ]

    @property
    def graph_obj(self) -> Voronoi:
        return self.__voronoi

    @property
    def region_qualified_dict(self):
        return self.__region_qualified_dict

    def __init__(self, xy_coord: np.ndarray, init_value=nan_arr()):
        super().__init__(xy_coord, init_value=init_value)
        # self.feature_set_value(self.KEY_CHORD, [np.nan, ])
        if self.valid_flag:
            self.__voronoi = Voronoi(points=xy_coord)
            self.__region_points = self.__voronoi.vertices
            self.__regions_all: Sequence[
                Sequence[int]] = self.graph_obj.regions
            self.__region_qualified_dict: Dict[
                int, Sequence[int]] = GraphUtil.region_qualified(
                    self.__region_points, self.__regions_all)
        else:
            self.__voronoi = None
            self.__region_points = None
            self.__regions_all = None
            self.__region_qualified_dict = None

    @staticmethod
    def concatenate_helper(arr_in):
        if len(arr_in) >= 2:
            return np.concatenate(arr_in)
        return np.asarray(arr_in)

    def feature_helper_valid(self, result: FeatureStore) -> FeatureStore:
        assert self.valid_flag
        chord_dist_regions_list: List[np.ndarray] = [
            GraphUtil.chord_distance_by_region(self.__region_points, region)
            for region in self.__region_qualified_dict.values()
        ]
        peri_by_region_list: List[float] = [
            GraphUtil.perimeter_by_region(self.__region_points, region)
            for region in self.__region_qualified_dict.values()
        ]
        area_by_region_list: List[float] = [
            GraphUtil.qhull_area_by_region(self.__region_points, region)
            for region in self.__region_qualified_dict.values()
        ]

        result[VoronoiFeature.KEY_AREA] = GraphFeature.validate_float_list(
            area_by_region_list)
        result[VoronoiFeature.KEY_PERI] = GraphFeature.validate_float_list(
            peri_by_region_list)
        chord_list = GraphFeature.validate_float_list(chord_dist_regions_list,
                                                      np.asarray([0.]))
        result[VoronoiFeature.KEY_CHORD] = self.concatenate_helper(chord_list)
        result[GraphFeature.KEY_GRAPH_OBJ] = self.graph_obj
        return result


class DelaunayFeature(GraphFeature):
    """
    Delaunay feature that calculates the perimeter and area of each simplices.
    """
    KEY_PERI: str = 'perimeter'
    KEY_AREA: str = 'area'

    __delaunay: Union[Delaunay, None]

    def init_value_curate(self):
        ...

    @staticmethod
    def is_coord_valid(xy_coord: np.ndarray):
        assert xy_coord.ndim == 2
        return xy_coord.shape[0] >= 4

    @property
    def feature_names(self):
        return [
            DelaunayFeature.KEY_PERI, DelaunayFeature.KEY_AREA,
            GraphFeature.KEY_GRAPH_OBJ
        ]

    @property
    def graph_obj(self) -> Delaunay:
        return self.__delaunay

    def __init__(self, xy_coord: np.ndarray, init_value=nan_arr()):
        super().__init__(xy_coord, init_value)
        self.__delaunay = Delaunay(
            points=xy_coord) if self.valid_flag else None

    def feature_helper_valid(self, result) -> FeatureStore:
        assert self.valid_flag
        regions_all: np.ndarray = self.graph_obj.simplices
        peri_by_region_list: List[float] = [
            GraphUtil.perimeter_by_region(self.xy_coord_all, region)
            for region in regions_all
        ]
        area_by_region_list: List[float] = [
            GraphUtil.qhull_area_by_region(self.xy_coord_all, region)
            for region in regions_all
        ]
        result[DelaunayFeature.KEY_PERI] = GraphFeature.validate_float_list(
            peri_by_region_list)
        result[DelaunayFeature.KEY_AREA] = GraphFeature.validate_float_list(
            area_by_region_list)
        result[GraphFeature.KEY_GRAPH_OBJ] = self.graph_obj
        return result


class MSTFeature(GraphFeature):
    """
    MST feature with edge length.
    """
    KEY_EDGE_LEN: str = 'edge_len'

    def init_value_curate(self):
        ...

    @staticmethod
    def is_coord_valid(xy_coord: np.ndarray):
        assert xy_coord.ndim == 2
        return xy_coord.shape[0] >= 3

    @property
    def feature_names(self):
        return [MSTFeature.KEY_EDGE_LEN, GraphFeature.KEY_GRAPH_OBJ]

    @property
    def graph_obj(self) -> Delaunay:
        return self.__mst

    def __init__(self, xy_coord: np.ndarray, init_value=nan_arr()):
        super().__init__(xy_coord, init_value)
        sqr_mat: np.ndarray = GraphUtil.dist_mat(self.xy_coord_all)
        # noinspection PyTypeChecker
        self.__mst = minimum_spanning_tree(csgraph=sqr_mat)

    def feature_helper_valid(self, result: FeatureStore):
        assert self.valid_flag
        dense_matrix: np.ndarray = self.graph_obj.toarray()
        result[MSTFeature.KEY_EDGE_LEN] = dense_matrix[dense_matrix > 0]
        result[GraphFeature.KEY_GRAPH_OBJ] = self.graph_obj
        return result


class NucleiFeature(GraphFeature):
    """
    Depends on the Voronoi feature. Calculates the number of valid voronoi regions (non-empty and no points of infinity
    distances) and the sum of their area.
    """
    KEY_AREA: str = 'area'
    KEY_VORONOI_SIZE: str = 'vor_size'

    def init_value_curate(self):
        ...

    @staticmethod
    def is_coord_valid(xy_coord: np.ndarray):
        return VoronoiFeature.is_coord_valid(xy_coord)

    @property
    def graph_obj(self) -> Any:
        return self.vor_feat.graph_obj

    def __init__(self, vor_feat: VoronoiFeature, init_value=nan_arr()):
        super().__init__(vor_feat.xy_coord_all, init_value=init_value)
        self.vor_feat = vor_feat

    @property
    def feature_names(self):
        return [NucleiFeature.KEY_AREA, NucleiFeature.KEY_VORONOI_SIZE]

    def feature_helper_valid(self, result: FeatureStore):
        assert self.valid_flag
        vor_size = len(self.vor_feat.region_qualified_dict)
        features: Dict[str, Sequence] = self.vor_feat.features
        sum_area = sum(features[VoronoiFeature.KEY_AREA])
        result[NucleiFeature.KEY_AREA] = sum_area
        result[NucleiFeature.KEY_VORONOI_SIZE] = vor_size
        return result


class KNNFeature(GraphFeature):
    """
    KNN feature - a helper feature to caluclate the RestrictedRadiusKNN below.
    Return the sorted knn distance table of all points for each given k for further query.
    """
    KEY_DIST: str = 'neighbor_dist'
    __sorted_neighbor_dist: np.ndarray

    @staticmethod
    def is_coord_valid(xy_coord: np.ndarray):
        assert xy_coord.ndim == 2
        return xy_coord.shape[0] >= 3

    @property
    def feature_names(self):
        return [KNNFeature.KEY_DIST]

    @property
    def ks(self):
        return self.__ks

    @property
    def dist_mat(self):
        return self.__dist_mat

    @property
    def graph_obj(self) -> Any:
        return self.dist_mat

    def init_value_curate(self):
        default_dist = {k: np.atleast_2d(nan_arr()) for k in self.ks}
        self._feature_set_value(self.KEY_DIST, default_dist)

    def __init__(self,
                 xy_coord: np.ndarray,
                 ks: Sequence[int],
                 init_value=nan_arr()):
        # very bad practice. coupling of init value with parameters after inst the parent class
        self.__ks = ks
        super().__init__(xy_coord, init_value=init_value)
        # N * N fully connected graph

        self.__dist_mat = GraphUtil.dist_mat(xy_coord)
        # Sort the distance matrix by columns in ascending order --> so the first k elements are the
        # k nearest neighbors
        self.__sorted_neighbor_dist = np.sort(self.dist_mat, axis=1)

    def feature_helper_valid(self, result):
        assert self.valid_flag
        knn_dist = {
            k: self.__sorted_neighbor_dist[:, 1:k + 1]
            for k in self.ks
        }
        result[KNNFeature.KEY_DIST] = knn_dist
        return result


class RestrictedRadiusKNNFeature(GraphFeature):
    """
    RestrictedRadiusKNNFeature calculated that given a radius, how many neighbors are within the radius of
    a given node.
    A set of different radius values are given.
    """
    KEY_RR: str = 'restricted_radius'

    @staticmethod
    def is_coord_valid(xy_coord: np.ndarray):
        return KNNFeature.is_coord_valid(xy_coord)

    @property
    def feature_names(self):
        return [RestrictedRadiusKNNFeature.KEY_RR]

    @property
    def rs(self):
        return self.__rs

    @property
    def graph_obj(self) -> Any:
        return self.__knn_feat.graph_obj

    def init_value_curate(self):
        r_count = {radius: np.nan for radius in self.rs}
        self._feature_set_value(self.KEY_RR, r_count)

    def __init__(self,
                 knn_feat: KNNFeature,
                 rs: Sequence[int],
                 init_value=nan_arr()):
        self.__rs = rs
        super().__init__(xy_coord=knn_feat.xy_coord_all, init_value=init_value)
        self.__knn_feat: KNNFeature = knn_feat

    def feature_helper_valid(self, result):
        assert self.valid_flag
        dist_mat = self.__knn_feat.dist_mat
        # -1 to remove self
        restricted_neighbor_count = {
            radius: (dist_mat <= radius).sum(axis=1) - 1
            for radius in self.rs
        }
        result[RestrictedRadiusKNNFeature.KEY_RR] = restricted_neighbor_count
        return result


class FeatureGenerator:
    """
    First order stats and more based on all features above.
    Mean, std, range, and disorder are calculated.
    """

    def init_value_curate(self):
        ...

    @property
    def xy_coord(self):
        return self.__xy_coord

    def __init__(self,
                 xy_coord: np.ndarray,
                 ks: Sequence[int] = (3, 5, 7),
                 rs: Sequence[int] = (10, 20, 30, 40, 50)):
        self.__xy_coord = xy_coord
        self._vor_feat = VoronoiFeature(self.__xy_coord)
        self._nuc_feat = NucleiFeature(self._vor_feat)

        self._del_feat = DelaunayFeature(self.__xy_coord)
        self._mst_feat = MSTFeature(self.__xy_coord)
        self._knn_feat = KNNFeature(self.__xy_coord, ks)
        self._rr_knn_feat = RestrictedRadiusKNNFeature(self._knn_feat, rs)

    def voronoi_feat_helper(self) -> Dict[str, float]:

        feat_base = self._vor_feat.features
        area_features_src = feat_base[VoronoiFeature.KEY_AREA]
        result = OrderedDict()
        result[_FeatName.VOR_AREA_STD] = np.std(area_features_src)
        result[_FeatName.VOR_AREA_MEAN] = np.mean(area_features_src)

        result[_FeatName.VOR_AREA_MIN_MAX] = GraphUtil.min_max(
            area_features_src)
        result[_FeatName.VOR_AREA_DISORDER] = GraphUtil.disorder(
            area_features_src)

        peri_src = feat_base[VoronoiFeature.KEY_PERI]
        result[_FeatName.VOR_PERI_STD] = np.std(peri_src)
        result[_FeatName.VOR_PERI_MEAN] = np.mean(peri_src)
        result[_FeatName.VOR_PERI_MIN_MAX] = GraphUtil.min_max(peri_src)
        result[_FeatName.VOR_PERI_DISORDER] = GraphUtil.disorder(peri_src)

        # might be nan --> the default value setting behavior for invalid cases should be refined later
        # i.e., set for each feature case by case. For now it all returns np.asarray([np.nan, ]),
        # which may not be suitable for some feature values
        chord_src = feat_base[VoronoiFeature.KEY_CHORD]
        result[_FeatName.VOR_CHORD_STD] = np.std(chord_src)
        result[_FeatName.VOR_CHORD_MEAN] = np.mean(chord_src)
        result[_FeatName.VOR_CHORD_MIN_MAX] = GraphUtil.min_max(chord_src)
        result[_FeatName.VOR_CHORD_DISORDER] = GraphUtil.disorder(chord_src)

        return result

    def del_feature_helper(self) -> Dict[str, float]:
        feat_base = self._del_feat.features
        result = OrderedDict()

        peri_src = feat_base[DelaunayFeature.KEY_PERI]
        # order todo
        result[_FeatName.DEL_PERI_MIN_MAX] = GraphUtil.min_max(peri_src)
        result[_FeatName.DEL_PERI_STD] = np.std(peri_src)
        result[_FeatName.DEL_PERI_MEAN] = np.mean(peri_src)
        result[_FeatName.DEL_PERI_DISORDER] = GraphUtil.disorder(peri_src)

        area_src = feat_base[DelaunayFeature.KEY_AREA]
        # order todo
        result[_FeatName.DEL_AREA_MIN_MAX] = GraphUtil.min_max(area_src)
        result[_FeatName.DEL_AREA_STD] = np.std(area_src)
        result[_FeatName.DEL_AREA_MEAN] = np.mean(area_src)
        result[_FeatName.DEL_AREA_DISORDER] = GraphUtil.disorder(area_src)
        return result

    def mst_feature_helper(self) -> Dict[str, float]:
        feat_base = self._mst_feat.features
        result = OrderedDict()

        edge_src = feat_base[MSTFeature.KEY_EDGE_LEN]
        # order todo
        result[_FeatName.MST_EDGE_MEAN] = np.mean(edge_src)
        result[_FeatName.MST_EDGE_STD] = np.std(edge_src)
        result[_FeatName.MST_EDGE_MIN_MAX] = GraphUtil.min_max(edge_src)
        result[_FeatName.MST_EDGE_DISORDER] = GraphUtil.disorder(edge_src)
        return result

    def nuclei_feat_helper(self) -> Dict[str, float]:
        feat_base = self._nuc_feat.features
        result = OrderedDict()

        result[_FeatName.NUC_SUM] = np.sum(feat_base[NucleiFeature.KEY_AREA])
        result[_FeatName.NUC_VOR_AREA] = feat_base[
            NucleiFeature.KEY_VORONOI_SIZE]
        result[_FeatName.NUC_DENSITY] = result[_FeatName.NUC_VOR_AREA] / (
            result[_FeatName.NUC_SUM] + eps)

        return result

    def knn_feat_helper(self) -> Dict[str, float]:
        feat_base = self._knn_feat.features
        result = OrderedDict()
        knn_dist: Dict[int, np.ndarray] = feat_base[KNNFeature.KEY_DIST]
        for k, dist_val in knn_dist.items():
            dist_val = np.atleast_2d(dist_val)
            dist_src = dist_val.sum(axis=1)
            result[f"{_FeatName.KNN_STD}{k}"] = np.std(dist_src)
            result[f"{_FeatName.KNN_MEAN}{k}"] = np.mean(dist_src)
            result[f"{_FeatName.KNN_DISORDER}{k}"] = GraphUtil.disorder(
                dist_src)
        return result

    def restricted_radius_feat_helper(self) -> Dict[str, float]:
        feat_base = self._rr_knn_feat.features
        result = OrderedDict()
        rr_count_dict: Dict[int, np.ndarray] = feat_base[
            RestrictedRadiusKNNFeature.KEY_RR]
        for k, count in rr_count_dict.items():
            count_src = count
            result[f"{_FeatName.RR_STD}{k}"] = np.std(count_src)
            result[f"{_FeatName.RR_MEAN}{k}"] = np.mean(count_src)
            result[f"{_FeatName.RR_DISORDER}{k}"] = GraphUtil.disorder(
                count_src)
        return result

    @LazyProperty
    def features(self):
        result = OrderedDict()
        vor_feat_set = self.voronoi_feat_helper()
        result.update(vor_feat_set)
        del_feat_set = self.del_feature_helper()
        result.update(del_feat_set)
        mst_feat_set = self.mst_feature_helper()
        result.update(mst_feat_set)
        nuc_feat_set = self.nuclei_feat_helper()
        result.update(nuc_feat_set)
        knn_feat_set = self.knn_feat_helper()
        result.update(knn_feat_set)
        rr_feat_set = self.restricted_radius_feat_helper()
        result.update(rr_feat_set)

        return result
