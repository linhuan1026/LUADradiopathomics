import numpy as np
from shapely.geometry import Polygon as SPolygon

from pathomics.matlab.flock_extractor.flock.core import FlockCore, FlockTyping, ClusterSkeleton
from pathomics.matlab.flock_extractor.flock.feature.implementation.cluster_level.cluster import FlockPolygons
from lazy_property import LazyProperty
from typing import Dict, Union, Sequence, Any

from pathomics.matlab.flock_extractor.flock.feature.util import GraphUtil, JsonDict
from pathomics.matlab.flock_extractor.flock.feature.implementation.flock_level.helper import FeatReduction

from pathomics.matlab.flock_extractor.flock.feature.implementation import FeatureSkeleton, PolygonFeature, FeatureGenerator
from pathomics.matlab.flock_extractor.flock.feature.implementation.flock_level import data_msg
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
eps = np.finfo(dtype=np.float32).eps


class PolygonContainer:
    """
    A helper class to calculate intersection of flock polygons and the coordinates of intersection regions.
    Use the Shapely package to calculate these stats.

    Polygon indices are given by the flock cluster indices.

    minimum_size_inclusive controls minimum required # of vertices in the polygon.
    Implicit requirement: minimum_size_inclusive >=3

    """
    __spolygon_cache_dict: Dict[int, SPolygon]
    __polygon_dict: Dict[int, Dict]
    __minimum_size_inclusive: int

    KEY_TARGET_IDX: str = 'target_id'
    KEY_INTER_AREA: str = 'intersection'
    KEY_INTER_CENTROID: str = 'inter_centroid'

    @property
    def spolygon_cache_dict(self):
        """
        A cache of Shapely polygon objects. Using Shapely for convenience to calculate intersection area and
        intersection region coordinates
        Returns:

        """
        return self.__spolygon_cache_dict

    # min ratio max ratio centroid pair
    @staticmethod
    def _init_spolygon(polygon_dict, minimum_size_inclusive):
        """
        Helper function. Generate and cache spolygon object for cluster that has sufficient vertices

        Args:
            polygon_dict: Dict of PolygonFeautres. Polygon indx --> polygon feature.
            minimum_size_inclusive:

        Returns:
            cluster id --> spolygon
        """
        spolygon_cache_dict = dict()
        for cluster_idx, polygon_data in polygon_dict.items():
            pts = polygon_data[PolygonFeature.KEY_CONTOUR]
            if not pts.shape[0] >= minimum_size_inclusive:
                continue
            spolygon_cache_dict[cluster_idx] = SPolygon(pts)
        return spolygon_cache_dict

    def _init_polygon_cache(self):
        """
        Generate and cache spolygon object into __spolygon_cache_dict
        Returns:

        """
        self.__spolygon_cache_dict = PolygonContainer._init_spolygon(
            self.__polygon_dict, self.__minimum_size_inclusive)

    def __init__(self, polygon_dict: Dict, minimum_size_inclusive: int):
        self.__minimum_size_inclusive = minimum_size_inclusive
        self.__polygon_dict = polygon_dict
        self._init_polygon_cache()

    @staticmethod
    def intersect_area(poly1: SPolygon, poly2: SPolygon):
        """
        For convenience I use shapely package. Hopefully this package should
        only be used within this closure.
        Args:
            poly1:
            poly2:

        Returns:

        """
        return poly1.intersection(poly2).area

    @staticmethod
    def intersect_area_coord(pts1, pts2):
        """
        Calculate intersection area by point coordinates of two polygons.
        Args:
            pts1:
            pts2:

        Returns:

        """
        poly1 = SPolygon(pts1)
        poly2 = SPolygon(pts2)
        return PolygonContainer.intersect_area(poly1, poly2)

    @staticmethod
    def intersect_by_idx_helper(src_idx, target_idx, spolygon_cache):
        """
        helper func to calculate intersection area by polygon index and given spolygon cache.
        Args:
            src_idx:
            target_idx:
            spolygon_cache:

        Returns:

        """
        if spolygon_cache.get(src_idx) is None or spolygon_cache.get(
                target_idx) is None:
            return 0.
        p1 = spolygon_cache[src_idx]
        p2 = spolygon_cache[target_idx]
        return PolygonContainer.intersect_area(p1, p2)

    @staticmethod
    def intersect_centroid(src_idx, target_idx, spolygon_cache):
        """
        Get the centroid of intersection region --> to build spatial graph of intersection regions
        Args:
            src_idx:
            target_idx:
            spolygon_cache:

        Returns:

        """
        if spolygon_cache.get(src_idx) is None or spolygon_cache.get(
                target_idx) is None:
            return None
        p1: SPolygon = spolygon_cache[src_idx]
        p2: SPolygon = spolygon_cache[target_idx]
        return p1.intersection(p2).centroid

    @staticmethod
    def intersect_pair_by_indices_static(src_idx, target_indices,
                                         spolygon_cache):
        """
        Returns intersection area of each pair of polygon between src_idx and all target_indices
        Args:
            src_idx: Specified source polygon
            target_indices: A list of target polygon indices to intersect
            spolygon_cache: Shapely polygon dict

        Returns:
            src_index -->  dict of <tgt_idx,
             intersect area,
             centroids>
             If there is no polygon or no intersection --> area = 0.
             intersection is either empty or none
        """
        result = {
            tgt: {
                PolygonContainer.KEY_TARGET_IDX:
                tgt,
                PolygonContainer.KEY_INTER_AREA:
                PolygonContainer.intersect_by_idx_helper(
                    src_idx, tgt, spolygon_cache),
                PolygonContainer.KEY_INTER_CENTROID:
                PolygonContainer.intersect_centroid(src_idx, tgt,
                                                    spolygon_cache)
            }
            for tgt in target_indices
        }
        return result

    def intersect_pair_by_indices(self, src_idx, target_indices):
        """
        for polygon src_idx, find the intersection of all polygon by given target indices
        Args:
            src_idx:
            target_indices:

        Returns:

        """
        return PolygonContainer.intersect_pair_by_indices_static(
            src_idx, target_indices, self.spolygon_cache_dict)


class FlockFeature(FeatureSkeleton):
    __flock_polygons: FlockPolygons
    __polygon_dict: Dict
    __dist_dict_self_excluded: Dict[int, Dict[int, Union[float, np.ndarray]]]
    __area_dict: Dict[int, Dict]

    KEY_INTERMEDIATE: str = 'intermediate_parent_node'

    KEY_TOTAL_INTERSECT: str = 'count_total_intersect'
    KEY_ABS_NZ_AREA_LIST: str = 'abs_area_list'
    KEY_ABS_NZ_AREA_PREFIX: str = 'abs_area_'

    KEY_INTERSECT_PORTION: str = 'count_portion_intersected'
    KEY_MAX_RATIO_PAIR: str = 'max_intersect_area_ratio'
    KEY_MIN_RATIO_PAIR: str = 'min_intersect_area_ratio'
    KEY_MAX_RATIO_COUNT_PREFIX: str = 'max_intersect_area_count_'
    KEY_MIN_RATIO_STAT_PREFIX: str = 'min_intersect_area_stat_'

    KEY_MAX_RATIO_OVER_CLUST_PREFIX: str = 'max_intersect_area_over_clust_'

    KEY_AREA_DICT: str = 'intermediate_area_dict'

    KEY_FLOCK_SIZE: str = 'flock_size_cluster_count'
    KEY_NUM_NUC_IN_CLUSTER: str = 'flock_num_nuc_in_cluster'
    KEY_NUC_DENSITY_OVER_SIZE: str = 'flock_nuc_density_over_size'
    KEY_NUC_NUM_PREF: str = 'flock_num_nuc_in_cluster_'
    KEY_NUC_DENSITY_PREF: str = 'flock_nuc_density_over_size_'

    KEY_FLOCK_ATTR: str = 'flock_other_attr'
    KEY_FLOCK_SPAN: str = 'span_var_dist_2_clust_cent'
    KEY_FLOCK_ATTR_PREF: str = 'flock_other_attr_'
    KEY_FLOCK_SPAN_PREF: str = 'span_var_dist_2_clust_cent_'

    KEY_PHENO_INTER: str = 'pheno_inter_type'
    KEY_PHENO_INTRA: str = 'pheno_intra_type'

    KEY_PHENO_INTER_DIFF_STAT_PREF: str = 'pheno_inter_diff_type_stat_'
    KEY_PHENO_INTRA_SAME_STAT_PREF: str = 'pheno_intra_same_type_stat_'

    KEY_PHENO_ENRICHMENT: str = 'pheno_enrichment'
    KEY_PHENO_ENRICHMENT_PREF: str = f"{KEY_PHENO_ENRICHMENT}_"

    KEY_PHENO_SPATIAL_GRAPH_PREF: str = 'pheno_spatial_graph_by_type_'

    KEY_OUT_INTERSECT: str = 'out_intersect'
    KEY_OUT_SPATIAL_INTER: str = 'out_spatial_intersect'
    KEY_OUT_SPATIAL_CLUSTER: str = 'out_spatial_cluster'
    KEY_OUT_POLYGON: str = 'out_polygon'
    KEY_OUT_SIZE: str = 'out_size'
    KEY_OUT_PHENO: str = 'out_phenotype'

    @property
    def flock_core(self) -> FlockCore:
        return self.__flock_core

    @property
    def flock_typing(self) -> FlockTyping:
        return self.__flock_typing

    @property
    def flock_polygons(self):
        return self.__flock_polygons

    @property
    def polygon_dict(self):
        return self.__polygon_dict

    @property
    def polygon_container(self):
        return self._polygon_container

    @property
    def num_neighbor_polygon(self) -> int:
        return self.__num_neighbor_polygon

    @property
    def num_neighbors_enrich(self) -> Sequence[int]:
        return self.__num_neighbors_enrich

    @property
    def dist_dict_self_excluded(self):
        return self.__dist_dict_self_excluded

    @property
    def area_dict(self):
        return self.__area_dict

    @staticmethod
    def cluster_dist_dict_self_excluded_valid_polygon(
            spatial_cluster_center, polygon_dict) -> Dict[int, Dict]:
        """
        Since some of clusters are removed (e.g. 2 nuclei only), the row index of numpy array of a distance matrix
        will not work. use Dict of Dict instead to construct distance matrix.
        Args:
            spatial_cluster_center:
            polygon_dict:

        Returns:
            Dict[int, Dict[int, float]]: cluster index --> <target cluster indx -> distance of src-target >
        """
        valid_cluster_id = np.asarray(list(polygon_dict.keys()))
        # valid_center = spatial_cluster_center[valid_cluster_id]
        dist_mat_all = FlockFeature.cluster_dist_mat_self_inf_helper(
            spatial_cluster_center)
        # dist_dict = {valid_idx: dist_mat_all[valid_idx, valid_cluster_id] for valid_idx in valid_cluster_id}
        dist_dict = {
            valid_idx: {
                idx: dist_mat_all[valid_idx, idx]
                for idx in valid_cluster_id
                if dist_mat_all[valid_idx, idx] != np.inf
            }
            for valid_idx in valid_cluster_id
        }
        return dist_dict

    @property
    def intersect_ratio_thresh(self):
        return self.__intersect_ratio_thresh

    def __init__(self,
                 flock_core: FlockCore,
                 flock_typing: FlockTyping,
                 num_neighbor_polygon: int = 30,
                 intersect_ratio_thresh: Sequence[float] = (0.1, 0.2, 0.3),
                 num_neighbors_enrich: Sequence[int] = (5, 10, 15)):
        """

        Args:
            flock_core:
            flock_typing:
            num_neighbor_polygon: Threshold of neighboring polygons for intersection calculation. Only the nearest
                k polygons defined here are included to calculate the intersection-based features
            intersect_ratio_thresh: Flock grades the intersection into three level: small/median/large, and each is
                given by a threshold of intersection area ratio (to the whole polygon area)
            num_neighbors_enrich: set of ks for knn phenotyping enrichment feature.  # of other phenotypes in k
                neighboring clusters.
        """
        self.__flock_core = flock_core
        self.__flock_typing = flock_typing
        self._register_polygons()
        spatial_cluster_center, _ = FlockCore.split_feat(
            flock_core.cluster_info_[FlockCore.KEY_CENTER],
            flock_core.feat_slice_id)

        self.__dist_dict_self_excluded: Dict[int, Dict[int, float]] = FlockFeature.\
            cluster_dist_dict_self_excluded_valid_polygon(spatial_cluster_center, self.polygon_dict)

        self._polygon_container = PolygonContainer(
            polygon_dict=self.polygon_dict,
            minimum_size_inclusive=self.flock_polygons.minimum_size_inclusive)
        self.__num_neighbor_polygon = num_neighbor_polygon
        self.__num_neighbors_enrich = num_neighbors_enrich
        self.__intersect_ratio_thresh = intersect_ratio_thresh
        # Nested dict of pair-wise polygon relationship
        self.__area_dict: Dict[int,
                               Dict[int,
                                    Dict]] = FlockFeature.area_dict_helper(
                                        self.dist_dict_self_excluded,
                                        self._polygon_container,
                                        self.num_neighbor_polygon)

    @staticmethod
    def __polygons_from_flock(flock_core: FlockCore) -> FlockPolygons:
        """
        Build the FlockPolygon object from flock_core. Convert each flock cluster to a polygon.
        Args:
            flock_core:

        Returns:

        """
        return FlockPolygons.build(flock_core)

    def _register_polygons(self):
        """
        Register and cache the polygon information derived from each flock cluster
        Returns:

        """
        self.__flock_polygons = FlockFeature.__polygons_from_flock(
            self.__flock_core)
        self.__polygon_dict = self.__flock_polygons.features

    @staticmethod
    def cluster_dist_mat_self_inf_helper(cluster_spatial_feat: np.ndarray):
        """
        Distance matrix of given feature points.
        Set the distance of a cluster to itself as infinity.
        Note: cannot be used directly to flock clusters as the cluster idx may not be consecutive after removal
        of cluster of one/two nuclei
        Args:
            cluster_spatial_feat:

        Returns:

        """
        dist_mat = GraphUtil.dist_mat(cluster_spatial_feat)
        assert dist_mat.ndim == 2

        assert dist_mat.shape[0] == dist_mat.shape[1]
        np.fill_diagonal(dist_mat, np.inf)
        return dist_mat

    @staticmethod
    def _dict_sort_by_value(dict_to_sort: Dict):
        """
        Sort dict by value. Helper function of knn query for dict based distance matrix.
        Args:
            dict_to_sort:

        Returns:

        """
        return {
            k: v
            for k, v in sorted(dict_to_sort.items(), key=lambda item: item[1])
        }

    @staticmethod
    def nearest_n_polygon(dist_dict_self_excluded: Dict[int, Dict],
                          num_neighbor: int = 30) -> Dict[int, np.ndarray]:
        """
        Find nearest n polygons to calculate intersection features. Return the corresponding index of nearest
        polygons. Polygon index is its flock cluster index.
        Args:
            dist_dict_self_excluded:
            num_neighbor:

        Returns:
            Dict: source polygon index --> np.ndarray of indices of nearest target polygons
        """
        # ascending order.
        # num_all_polygons = len(list(dist_dict_self_excluded.values())[0])
        # if num_neighbor is over max polygon num, need to
        # num_neighbor = min(num_all_polygons - 1, num_neighbor)
        sorted_dist = {
            cluster_idx: FlockFeature._dict_sort_by_value(dist_dict_row)
            for cluster_idx, dist_dict_row in dist_dict_self_excluded.items()
        }

        index_dist = {
            cluster_idx:
            np.asarray(list(sorted_dist_row.keys()))[:num_neighbor]
            for cluster_idx, sorted_dist_row in sorted_dist.items()
        }
        return index_dist

    @staticmethod
    def _interm_dict(out_dict: Dict):
        """
        Prepare the intermediate results.
        Args:
            out_dict:

        Returns:

        """
        interm = out_dict.get(FlockFeature.KEY_INTERMEDIATE, dict())
        out_dict[FlockFeature.KEY_INTERMEDIATE] = interm
        return interm

    @staticmethod
    def area_dict_helper(
            dist_dict_self_excluded: Dict[int, Dict[int, Union[float,
                                                               np.ndarray]]],
            polygon_container: PolygonContainer,
            num_neighbor: int = 30):
        """
        Calculate the intersection area and other intermediate values between each pair of polygons.
        These pairs are not ordered and are not unique (e.g., pair <a,b> and <b,a> coexists. The unique pairs
        are further processed in intersect_point_set method.
        Args:
            dist_dict_self_excluded: Dict based distance matrix between clusters of non-consecutive cluster idx
            polygon_container: PolygonContainer objects with Shapely polygon objects for intersection area/region
            num_neighbor: threshold of # of neighboring clusters that are allowed for intersection feature calculation

        Returns:
            Dict[int, Dict[int, Dict]]: First level: source polygon index --> Dict of compounds of target
            polygon. Second level: target polygon index --> Dict of intersection area and other intermediate values.
            Third level: feature/intermediate value name (str) --> values
            Contains: target_id, intersection area, intersection centroids.
        """
        dist_dict_idx: Dict[int, np.ndarray] = FlockFeature.nearest_n_polygon(
            dist_dict_self_excluded, num_neighbor)
        result: Dict[int, Dict[int, Dict[str, float]]] = dict()
        for cluster_idx, each_row in dist_dict_idx.items():
            # Note: each_row may have
            result[cluster_idx] = polygon_container.intersect_pair_by_indices(
                cluster_idx, each_row)
        if len(result) == 0:
            logger.warning(data_msg.MSG_EMPTY_AREA_DICT)
        return result

    @staticmethod
    def __valid_cluster_size(area_dict: Dict):
        return max(len(area_dict), 1)

    @staticmethod
    def _validate_dict(dict_in: Dict, msg):
        if len(dict_in) == 0:
            raise ValueError(msg)
        return dict_in

    @staticmethod
    def _validate_area_dict(area_dict, extra_msg: str = ''):
        return FlockFeature._validate_dict(
            area_dict, data_msg.MSG_EMPTY_AREA_DICT + extra_msg)

    @LazyProperty
    def intersect_feature_helper(self) -> Dict:
        """
        Intersection feature.
        Returns:

        """
        FlockFeature._validate_area_dict(self.area_dict)
        result = dict()

        # todo reduce
        # for each polygon, count the number of other polygons intersected with it
        abs_intersect_area = [
            np.count_nonzero([
                v[PolygonContainer.KEY_INTER_AREA]
                for k, v in single_polygon.items()
            ]) for single_polygon in self.area_dict.values()
        ]
        # statistics of intersect counts for all polygons
        result.update(
            FeatReduction.statisti_reduce(np.asarray(abs_intersect_area),
                                          FlockFeature.KEY_ABS_NZ_AREA_PREFIX))

        # sum of all intersected counts across all polygons
        total_intersect = sum(abs_intersect_area)
        result[FlockFeature.KEY_TOTAL_INTERSECT] = total_intersect
        # polygon count --> use >3 only
        # use size of dist_mat or area_dict if you need all clusters regardless of whether polygons are formed.

        # number of polygons
        polygon_count = len(self._polygon_container.spolygon_cache_dict)
        # ratio of intersect count over number of polygons --> can be greater than one
        portion_intersect_ratio = total_intersect / (polygon_count + eps)
        result[FlockFeature.KEY_INTERSECT_PORTION] = portion_intersect_ratio
        # for polygon i and all polygon intersecting with i, the intersection over area of i
        left_intersect = {
            idx: [
                v[PolygonContainer.KEY_INTER_AREA] /
                (eps + self.polygon_dict[idx][PolygonFeature.KEY_POLYGON_AREA])
                for k, v in single_polygon.items()
            ]
            for idx, single_polygon in self.area_dict.items()
        }
        # for polygon i and all polygon j intersecting with i, the intersection over each j
        right_intersect = {
            idx: [
                v[PolygonContainer.KEY_INTER_AREA] /
                (eps + self.polygon_dict[v[PolygonContainer.KEY_TARGET_IDX]][
                    PolygonFeature.KEY_POLYGON_AREA])
                for k, v in single_polygon.items()
            ]
            for idx, single_polygon in self.area_dict.items()
        }
        assert left_intersect.keys() == right_intersect.keys()
        # the max ratio value of left_intersect and right_intersect
        max_ratio = {
            k: [
                max(ll, rr)
                for ll, rr in zip(left_intersect[k], right_intersect[k])
            ]
            for k in left_intersect.keys()
        }

        # count of ratio that is above all thresholds in self.intersect_ratio_thresh, which is
        # defined as intersection that is small/median/large

        max_ratio_threshed_count = FeatReduction.ratio_reduce(
            max_ratio,
            self.intersect_ratio_thresh,
            name_pref=FlockFeature.KEY_MAX_RATIO_COUNT_PREFIX,
            denominator=1.)
        result.update(max_ratio_threshed_count)

        valid_cluster_size = FlockFeature.__valid_cluster_size(self.area_dict)
        # same as max_ratio_threshed_count, but normalized by the number of clusters
        max_ratio_threshed_ratio = FeatReduction.ratio_reduce(
            max_ratio,
            self.intersect_ratio_thresh,
            name_pref=FlockFeature.KEY_MAX_RATIO_OVER_CLUST_PREFIX,
            denominator=valid_cluster_size)
        result.update(max_ratio_threshed_ratio)

        # similar to max ratio but the min value
        min_ratio = {
            k: [
                min(ll, rr)
                for ll, rr in zip(left_intersect[k], right_intersect[k])
            ]
            for k in left_intersect.keys()
        }
        # todo reduce

        min_ratio_flatten = FeatReduction.flattened_ratio(min_ratio,
                                                          positive_only=True)
        min_ratio_stat = FeatReduction.statisti_reduce(
            min_ratio_flatten,
            name_prefix=FlockFeature.KEY_MIN_RATIO_STAT_PREFIX)
        result.update(min_ratio_stat)
        # todo reduce
        intermediate_result = FlockFeature._interm_dict(result)
        intermediate_result[
            FlockFeature.KEY_ABS_NZ_AREA_LIST] = abs_intersect_area
        intermediate_result[FlockFeature.KEY_MAX_RATIO_PAIR] = max_ratio
        intermediate_result[FlockFeature.KEY_MIN_RATIO_PAIR] = min_ratio
        intermediate_result[FlockFeature.KEY_AREA_DICT] = self.area_dict
        return result

    @staticmethod
    def unique_intersect_pair(area_dict: Dict[int, Dict[int, Dict[str, Any]]]):
        """
        Find unique pairs of polygons that intersect with each other in area_dict (see area_dict_helper)

        Args:
            area_dict:

        Returns:

        """
        idx_pair_set = set()
        output_idx_list = []
        for src_idx, src_polygon_dict in area_dict.items():
            for target_idx, target_polygon_dict in src_polygon_dict.items():
                pair1 = src_idx, target_idx
                pair2 = target_idx, src_idx
                if pair1 in idx_pair_set or pair2 in idx_pair_set:
                    # already registered
                    continue
                idx_pair_set.add(pair1)
                idx_pair_set.add(pair2)
                inter_area = target_polygon_dict[
                    PolygonContainer.KEY_INTER_AREA]
                inter_centroid = target_polygon_dict[
                    PolygonContainer.KEY_INTER_CENTROID]
                if inter_area == 0 or inter_centroid is None or len(
                        inter_centroid.coords) == 0:
                    continue
                output_idx_list.append(pair1)
        return output_idx_list

    @staticmethod
    def intersect_point_set(area_dict):
        """
        Centroids of intersection region between unique pairs of intersected polygons
        Args:
            area_dict:

        Returns:
            np.ndarray: the centroids
        """
        assert len(area_dict) > 0
        unique_idx = FlockFeature.unique_intersect_pair(area_dict)
        key_inter = PolygonContainer.KEY_INTER_CENTROID
        row_points = [
            np.concatenate(area_dict[src][tgt][key_inter].coords.xy)
            for src, tgt in unique_idx
        ]
        # out_arr = np.vstack(row_points) ##ori
        try:
            out_arr = np.vstack(row_points)
        except:
            out_arr = np.array([[0, 0], [0, 0]])
        return out_arr

    @LazyProperty
    def spatial_feature_by_intersect(self) -> Dict:
        """
        Spatial low-level graph features of the intersection region
        Returns:

        """
        area_dict = FlockFeature._validate_area_dict(self.area_dict)
        point_sets = FlockFeature.intersect_point_set(area_dict)
        return FeatureGenerator(point_sets).features

    @LazyProperty
    def spatial_feature_by_cluster(self) -> Dict:
        """
        Spatial low-level graph features of the flock clusters
        Returns:

        """
        center_all_dim = self.flock_core.cluster_info_[FlockCore.KEY_CENTER]
        slice_id = self.flock_core.feat_slice_id
        spatial_center, _ = FlockCore.split_feat(center_all_dim, slice_id)
        return FeatureGenerator(spatial_center).features

    @LazyProperty
    def size_feature(self):
        """
        Features of nuclei count and density
        Returns:

        """
        result = dict()

        polygon_dict = self.polygon_dict
        flock_size = len(polygon_dict)
        result[FlockFeature.KEY_FLOCK_SIZE] = flock_size

        nuc_num = [
            v[PolygonFeature.KEY_NUM_NUC] for v in polygon_dict.values()
        ]
        nuc_num_stat = FeatReduction.statisti_reduce(
            nuc_num, FlockFeature.KEY_NUC_NUM_PREF)
        result.update(nuc_num_stat)

        nuc_density = [n / (eps + flock_size) for n in nuc_num]
        nuc_density_stat = FeatReduction.statisti_reduce(
            nuc_density, FlockFeature.KEY_NUC_DENSITY_PREF)
        result.update(nuc_density_stat)

        intermediate_result = FlockFeature._interm_dict(result)
        intermediate_result[FlockFeature.KEY_NUM_NUC_IN_CLUSTER] = nuc_num
        intermediate_result[
            FlockFeature.KEY_NUC_DENSITY_OVER_SIZE] = nuc_density
        return result

    @staticmethod
    def _validate_polygon_dict(area_dict):
        return FlockFeature._validate_dict(area_dict,
                                           data_msg.MSG_EMPTY_POLYGON_DICT)

    @LazyProperty
    def cluster_polygon_feature(self):
        """
        Polygon-level features: std of distance to spatial centers. Likewise, the std of distance of other attrs
        to the corresponding centers.
        Note: the feature vector for clustering is the concatenation of spatial coord and other attrs.
        Hence, the cluster center has a spatial center and an other-attr center
        Returns:

        """
        result = dict()

        polygon_dict = FlockFeature._validate_polygon_dict(self.polygon_dict)
        # todo reduce
        flock_attr = [
            v[PolygonFeature.KEY_ATTR_STD] for v in polygon_dict.values()
        ]
        flock_attr_stat = FeatReduction.statisti_reduce(
            flock_attr, FlockFeature.KEY_FLOCK_ATTR_PREF)
        result.update(flock_attr_stat)
        # todo reduce
        flock_span = [
            v[PolygonFeature.KEY_DIST_STD] for v in polygon_dict.values()
        ]
        flock_span_stat = FeatReduction.statisti_reduce(
            flock_span, FlockFeature.KEY_FLOCK_SPAN_PREF)
        result.update(flock_span_stat)
        intermediate_result = FlockFeature._interm_dict(result)
        intermediate_result[FlockFeature.KEY_FLOCK_ATTR] = flock_attr
        intermediate_result[FlockFeature.KEY_FLOCK_SPAN] = flock_span
        return result

    @staticmethod
    def phenotype_intersect(flock_typing, area_dict: Dict, phenotype: int):
        """
        Helper function
        Calculate the intra-type and inter-type statistics of unique pairs of intersected polygons (i.e. flock clusters)
        Args:
            flock_typing:
            area_dict:
            phenotype: specified input phenotype to compare

        Returns:
            Tuple[np.ndarray, np.ndarray, float]: For each unique pair of intersected polygons:
            Intra-type --> boolean flag of those share the exact same phenotype to the given phenotype
            type_a == type_b == phenotype
            Inter-type --> boolean flag of those with different phenotype to the given input:
            type_a != phenotype and type_b != phenotype
            total number of cluster
        """
        unique_idx = FlockFeature.unique_intersect_pair(area_dict)
        idx_2_label = flock_typing.cluster_info_[ClusterSkeleton.KEY_LABEL]
        type_cluster_count = len(
            flock_typing.cluster_info_[ClusterSkeleton.KEY_MEMBER][phenotype])
        intra_type_same = np.asarray([
            idx_2_label[src_ind] == phenotype
            and phenotype == idx_2_label[tgt_ind]
            for src_ind, tgt_ind in unique_idx
        ])
        inter_type_diff = np.asarray([
            idx_2_label[src_ind] != phenotype
            and phenotype != idx_2_label[tgt_ind]
            for src_ind, tgt_ind in unique_idx
        ])
        return intra_type_same, inter_type_diff, type_cluster_count

    # @staticmethod
    # def enrichment_by_neighbor(enrichment: Union[Sequence[float], np.ndarray],
    #                            n_neighbors: Union[Sequence[int], np.ndarray],
    #                            name_pref: str):
    #     result = dict()
    #     for nn in n_neighbors:
    #         name_pref_nn = f"{name_pref}n_{nn}"
    #         curr_enrich: Dict = FeatReduction.statisti_reduce(enrichment, name_pref_nn)
    #         assert curr_enrich.keys() not in result.keys()
    #         result.update(result)
    #     return result

    @staticmethod
    def enrichment_helper(idx_2_label: np.ndarray,
                          ind_exclude_self: np.ndarray,
                          n_neighbors: Union[Sequence[int], np.ndarray]):
        """
        Helper function. Enrichment of phenotypes
        Count the number of different phenotypes in the given neighborhood (k nearest neighbors)
        Args:
            idx_2_label: map the cluster index to the cluster label, given by phenotype typing
            ind_exclude_self: indices of the neighbors of each vertex. The index of the vertex itself is excluded
            n_neighbors: set of N

        Returns:
            Dict: first-order and other statistics of the enrichment counts
        """
        n_neighbors = np.asarray(n_neighbors)
        assert np.count_nonzero(n_neighbors) == n_neighbors.shape[0]
        assert np.max(n_neighbors) >= n_neighbors.shape[0]
        result = dict()
        enrichment = []
        for cluster_idx, each_row_of_neighbors in enumerate(ind_exclude_self):
            phenotype_of_curr_clust = idx_2_label[cluster_idx]
            enrich_curr = sum([
                idx_2_label[x] != phenotype_of_curr_clust
                for x in each_row_of_neighbors
            ])
            enrichment.append(enrich_curr)
        name_pref = f"{FlockFeature.KEY_PHENO_ENRICHMENT_PREF}"
        # FlockFeature.enrichment_by_neighbor(enrichment, n_neighbors, name_pref)
        result_nn = FeatReduction.statisti_reduce(enrichment, name_pref)
        result.update(result_nn)
        return result

    @staticmethod
    def phenotype_nn_enrichment(flock_typing: FlockTyping,
                                n_neighbors: Union[Sequence[int], np.ndarray]):
        """
        Enrichment of phenotypes
        Count the number of different phenotypes in the given neighborhood (k nearest neighbors)
        Args:
            flock_typing: Phenotyping clustering information
            n_neighbors: A set of different k for k nearest neighbors

        Returns:

        """
        cluster_centers = flock_typing.spatial_feat
        dist_mat = FlockFeature.cluster_dist_mat_self_inf_helper(
            cluster_centers)
        idx_2_label = flock_typing.cluster_info_[ClusterSkeleton.KEY_LABEL]
        ind_exclude_self = np.argsort(dist_mat)[:, :
                                                -1]  # exclude inf at the end
        # edge case --> only one cluster --> inc_exclude_self could be empty
        result = FlockFeature.enrichment_helper(idx_2_label, ind_exclude_self,
                                                n_neighbors)
        return result

    @LazyProperty
    def phenotype_feature_helper(self):
        """
        Phenotype features
        Returns:

        """
        area_dict = self.area_dict
        flock_typing: FlockTyping = self.flock_typing
        unique_labels = self.flock_typing.cluster_info_[
            ClusterSkeleton.KEY_CLUSTER_LABEL_SET]

        result = dict()
        # enrichment is the count of other phenotypes around a given flock cluster (how divergent the phenotypes
        # are within a local region)
        enrich_out = FlockFeature.phenotype_nn_enrichment(
            flock_typing, self.num_neighbors_enrich)
        result.update(enrich_out)
        for phenotype in unique_labels:
            # i would avoid using a == b == c
            # inter and intra- phenotype statstics - counts and ratio
            intra_type_same, inter_type_diff, type_cluster_count = FlockFeature.phenotype_intersect(
                flock_typing, area_dict, phenotype=phenotype)
            intra_feat = FeatReduction.phenotype_intersect_reduce(
                intra_type_same, type_cluster_count,
                FlockFeature.KEY_PHENO_INTRA_SAME_STAT_PREF, phenotype)
            result.update(intra_feat)
            inter_feat = FeatReduction.phenotype_intersect_reduce(
                intra_type_same, type_cluster_count,
                FlockFeature.KEY_PHENO_INTER_DIFF_STAT_PREF, phenotype)
            result.update(inter_feat)

            # spatial
            cluster_ind_by_type = flock_typing.cluster_info_[
                ClusterSkeleton.KEY_MEMBER][phenotype]
            assert len(cluster_ind_by_type) > 0
            graph_coords_xy = flock_typing.spatial_feat[cluster_ind_by_type]
            name_pref = FlockFeature.KEY_PHENO_SPATIAL_GRAPH_PREF
            stat_key = f"{name_pref}{phenotype}"

            # at least 3 nodes
            result[stat_key] = FeatureGenerator(graph_coords_xy).features
            # todo double check here -- no time
            # if graph_coords_xy.shape[0] >= 3:
            #     # breakpoint()
            #
            #     result[stat_key] = FeatureGenerator(graph_coords_xy).features
            # else:
            #     # todo double check --> without this line the 'out_phenotype' feature may yield features
            #     # todo of different sizes.
            #     # This happens in previous rounds of feature generation, and I hardcoded in data reading procedure
            #     # to curate the feature vector size
            #     result[stat_key] = np.nan
        return result

    @LazyProperty
    def features(self) -> Dict:
        """

        Returns:
            Dict: Intersection feature, spatial feature of intersection region, spatial feature of flock,
            nuclei number and density (size of cluster), flock polygon features, and phenotype features
        """
        result = dict()

        result[FlockFeature.KEY_OUT_INTERSECT] = self.intersect_feature_helper
        result[FlockFeature.
               KEY_OUT_SPATIAL_INTER] = self.spatial_feature_by_intersect
        result[FlockFeature.
               KEY_OUT_SPATIAL_CLUSTER] = self.spatial_feature_by_cluster
        result[FlockFeature.KEY_OUT_SIZE] = self.size_feature
        result[FlockFeature.KEY_OUT_POLYGON] = self.cluster_polygon_feature

        result[FlockFeature.KEY_OUT_PHENO] = self.phenotype_feature_helper
        return result

    @staticmethod
    def _flatten(in_dict: Dict[str, Any], excluded_keys: Sequence[str]):
        """
        Flatten the nested dict into plain list of feature vectors.
        Args:
            in_dict: Input nested dict (e.g. the dict flock feature)
            excluded_keys: keys of dict to ignore (e.g., intermediate results)

        Returns:
            Tuple[List[str], List[float]]: corresponding keys and list of feature values
        """
        out_keys = []
        out_vals = []
        for k, v in in_dict.items():
            if k in excluded_keys:
                continue
            if not isinstance(v, Dict):
                sub_keys = [k]
                sub_vals = [v]
                # out_vals.append(v)
            else:
                sub_keys, sub_vals = FlockFeature._flatten(v, excluded_keys)
            out_keys += sub_keys
            out_vals += sub_vals
        return out_keys, out_vals

    @LazyProperty
    def flattened_feature(self):
        """
        Convert dict feature into list of feature names and feature values. Note: for the current implementation
        there may be duplicate feature names in the output as the procedure omit the nested structure of the dict.
        For instance, the area of polygon 0 and polygon 1 are all named "feature_polygon" after flattening process,
        while their difference are shown in the nested dict by different parent keys (e.g., 0 vs. 1 as parent keys)
        Returns:
            Tuple[List[str], List[float]]: corresponding keys and list of feature values
        """
        return FlockFeature._flatten(self.features,
                                     FlockFeature.KEY_INTERMEDIATE)

    @staticmethod
    def _serializable_helper(in_dict: Dict[str, Any],
                             excluded_keys: Sequence[str]):
        """
        Helper function. Make each dict in the nested dict input serializable.
        Args:
            in_dict:
            excluded_keys:

        Returns:

        """
        out_dict = JsonDict()
        for k, v in in_dict.items():
            if k in excluded_keys:
                continue
            if not isinstance(v, Dict):
                out_dict[k] = v
            else:
                out_dict[k] = FlockFeature._serializable_helper(
                    v, excluded_keys)
        return out_dict

    @LazyProperty
    def to_json_serializable(self):
        """
        Discard objects that cannot be encoded into json, or converting numpy data into basic types,
        so that the json-form of the flock feature can be written to the disk for debugging purposes.
        """
        return FlockFeature._serializable_helper(self.features,
                                                 FlockFeature.KEY_INTERMEDIATE)

    @staticmethod
    def _flatten_nested(in_dict: Dict[str, Any], excluded_keys: Sequence[str]):
        """
        Flatten the nested dict into plain list of feature vectors.
        Seems to be a duplicate of _flatten that I forgot to remove
        Args:
            in_dict: Input nested dict (e.g. the dict flock feature)
            excluded_keys: keys of dict to ignore (e.g., intermediate results)

        Returns:
            Tuple[List[str], List[float]]: corresponding keys and list of feature values
        """
        out_keys = []
        out_vals = []
        for k, v in in_dict.items():
            if k in excluded_keys:
                continue
            if not isinstance(v, Dict):
                sub_keys = [k]
                sub_vals = [v]
                out_vals.append(v)
            else:
                sub_keys, sub_vals = FlockFeature._flatten(v, excluded_keys)
            out_keys += sub_keys
            out_vals += sub_vals
        return out_keys, out_vals