from scipy.spatial import ConvexHull
from scipy.spatial.distance import euclidean, squareform, pdist
import numpy as np
from collections import OrderedDict
from typing import Sequence, Tuple, Dict, Callable, List
from copy import deepcopy


eps = np.finfo(float).eps


class GraphUtil:
    """
    Utility functions for graph features.
    """

    @staticmethod
    def iqr_outlier(coord_single: np.ndarray) -> Tuple[float, float]:
        """
        Outlier detection by interquartile range
        Args:
            coord_single: coordinate value of a single dimension (e.g. all xs, or all ys in 2d plane)

        Returns:
            Low and High range given by iqr-based outlier detection.
        """
        q1 = np.percentile(coord_single, 25)
        q3 = np.percentile(coord_single, 75)
        iqr = q3 - q1
        high_range = q3 + 1.5 * iqr
        low_range = q1 - 1.5 * iqr
        return low_range, high_range

    @staticmethod
    def outlier_of_regions(xy_coord_all: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """

        Args:
            xy_coord_all: x and y coordiante in 2d array. N * 2

        Returns:
            low and high range for x and y dim correspondingly.
        """
        x_low, x_high = GraphUtil.iqr_outlier(xy_coord_all[:, 0])
        y_low, y_high = GraphUtil.iqr_outlier(xy_coord_all[:, 1])
        return (x_low, x_high), (y_low, y_high)

    @staticmethod
    def is_region_valid(region_single: Sequence[int]) -> bool:
        """
        Check whether region (list of index to xy_coords of voronoi) are valid.
        The region should be non-empty and contains all non-negative values. -1 in indices in qhull lib means
        the vertices is at inf.
        Args:
            region_single:

        Returns:
            whether the region is valid
        """
        has_point_at_inf = any(np.asarray(region_single) < 0)
        empty_region = len(region_single) <= 0
        return not (has_point_at_inf or empty_region)

    @staticmethod
    def coord_in_range(coord_val_single: np.ndarray, low: float, high: float) -> bool:
        """
        Check whether coords of a single dimension are all in the range
        Args:
            coord_val_single:
            low:
            high:

        Returns:

        """
        return all(coord_val_single <= high) and all(coord_val_single >= low)

    @staticmethod
    def region_validation(xy_coord_all: np.ndarray, region: Sequence[int],
                          check_range: bool = True,
                          x_low: float = np.NINF,
                          x_high: float = np.inf,
                          y_low: float = np.NINF,
                          y_high: float = np.inf):
        """
        Further Check whether a non-empty region is valid by limiting its min/max coordinates.
         This is mostly used in validating voronoi
        regions, as voronoi may generates points that is infinite away from the center.
        Skip the check if check_range is False
        Args:
            xy_coord_all:
            region:
            check_range:
            x_low:
            x_high:
            y_low:
            y_high:

        Returns:
            bool: False if the region is empty or have infinity-distance points
        """
        valid_flag = GraphUtil.is_region_valid(region)
        if (not valid_flag) or (not check_range):
            return valid_flag

        region_points = xy_coord_all[region]
        x_in_range = GraphUtil.coord_in_range(region_points[:, 0], x_low, x_high)
        y_in_range = GraphUtil.coord_in_range(region_points[:, 1], y_low, y_high)
        return x_in_range and y_in_range

    @staticmethod
    def region_qualified_helper(xy_coord_all: np.ndarray, regions_all: Sequence[Sequence[int]],
                                check_range: bool,
                                x_low: float = np.NINF,
                                x_high: float = np.inf,
                                y_low: float = np.NINF,
                                y_high: float = np.inf):
        """
        Helper function of region_qualified. Check whether regions are valid in all regions_all input.
        A region is invalid if the region is empty or have infinity-distance points.
        Ignore the infinity-distance points if check_range is set to False
        Args:
            xy_coord_all:
            regions_all:
            check_range:
            x_low:
            x_high:
            y_low:
            y_high:

        Returns:
            Dict: keys are valid regions idx. Values are the corresponding regions.
        """
        odict = OrderedDict()
        for idx, region in enumerate(regions_all):
            is_valid = GraphUtil.region_validation(xy_coord_all, region, check_range, x_low, x_high, y_low, y_high)
            if is_valid:
                odict[idx] = region
        return odict

    @staticmethod
    def region_qualified(xy_coord_all: np.ndarray, regions_all: Sequence[Sequence[int]]) -> Dict[int, Sequence[int]]:
        """
        A region is invalid if the region is empty or have infinity-distance points if there are at least 3 regions.
        Otherwise only check if it is empty.
        Args:
            xy_coord_all:
            regions_all:

        Returns:

        """
        (x_low, x_high), (y_low, y_high) = GraphUtil.outlier_of_regions(xy_coord_all)
        region_dict = GraphUtil.region_qualified_helper(xy_coord_all, regions_all, True,
                                                        x_low, x_high, y_low, y_high)
        if len(region_dict) <= 2:
            region_dict = GraphUtil.region_qualified_helper(xy_coord_all, regions_all, False)
        return region_dict

    @staticmethod
    def coord_qhull_sort(xy_coord_of_region: np.ndarray, do_sort: bool = False):
        """
        Make sure the order of given coordinates are the clockwise/counter-clockwise of the polygon border
        Args:
            xy_coord_of_region:
            do_sort: For simplification, if True feed it into ConvexHull and return the sorted vertices from the
                ConvexHull object

        Returns:

        """
        coord_work = xy_coord_of_region
        if do_sort:
            coord_work = ConvexHull(points=coord_work).vertices
        return coord_work

    @staticmethod
    def chord_by_coords(xy_coord_of_region: np.ndarray):
        """
        Helper function. Chord length (point distance to each other points)
        Args:
            xy_coord_of_region:

        Returns:

        """
        return pdist(xy_coord_of_region)

    @staticmethod
    def chord_distance_by_region(xy_coord_all, region: Sequence[int]) -> np.ndarray:
        """
        Chord length (point distance to each other points)
        Args:
            xy_coord_all:
            region:

        Returns:

        """
        region_coord = xy_coord_all[region]
        return GraphUtil.chord_by_coords(region_coord)

    @staticmethod
    def perimeter_by_coords(xy_coord_of_region: np.ndarray,
                            do_sort: bool = False,
                            dist_func: Callable = euclidean) -> float:
        """
        Calculate perimeter of the region polygon.
        Args:
            xy_coord_of_region: N * 2 coordinates
            do_sort: whether explicitly sort the input by convexhull.
            dist_func:

        Returns:

        """
        coord_work = GraphUtil.coord_qhull_sort(xy_coord_of_region, do_sort)
        # first/last
        coord_work_zig = np.concatenate([coord_work, coord_work[0][None, ]])[1:]
        perimeter = sum([dist_func(x, y) for x, y in zip(coord_work, coord_work_zig)])
        return perimeter

    @staticmethod
    def perimeter_by_region(xy_coord_all, region: Sequence[int]) -> float:
        """
        Calculate perimeter of the region polygon.
        Args:
            xy_coord_all:
            region:

        Returns:

        """
        region_coord = xy_coord_all[region]
        return GraphUtil.perimeter_by_coords(region_coord)

    @staticmethod
    def qhull_area_by_coords(xy_coord_of_region: np.ndarray) -> float:
        """
        Wrapper to generate convex hull objects. For now we use Qhull
        Args:
            xy_coord_of_region:

        Returns:

        """
        return ConvexHull(points=xy_coord_of_region).area

    @staticmethod
    def qhull_area_by_region(xy_coord_all, region: Sequence[int]) -> float:
        """
        Wrapper to generate convex hull objects. For now we use Qhull
        Args:
            xy_coord_all:
            region:

        Returns:

        """
        region_coord = xy_coord_all[region]
        return ConvexHull(points=region_coord).area

    @staticmethod
    def disorder_helper(std, mean):
        """
        calculate disorder (Graph feature metric)
        Args:
            std:
            mean:

        Returns:

        """
        return 1 - mean / (mean + std + eps)

    @staticmethod
    def disorder(src_data: Sequence[float]):
        std = np.std(src_data)
        mean = np.mean(src_data)
        return GraphUtil.disorder_helper(std, mean)

    @staticmethod
    def min_max(src_data: Sequence[float]):
        """
        calculate the range of the input array
        Args:
            src_data:

        Returns:

        """
        return np.min(src_data) / (np.max(src_data) + eps)

    @staticmethod
    def dist_mat(xy_coord: np.ndarray):
        """
        Get the distance matrix of the given coordinates. This is the adjacency matrix of a fully connected graph
        of the input points.
        Args:
            xy_coord:

        Returns:

        """
        return squareform(pdist(xy_coord))


class JsonDict(dict):
    """
    Explicit curation before any other extra JSONencoder. SImply convert numpy types to base types that
    can be encoded by the default JsonEncoder
    """

    @staticmethod
    def curate(obj):
        """
        based on Numpyencoder
        Args:
            obj:

        Returns:

        """

        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_,)):
            return bool(obj)

        elif isinstance(obj, (np.void,)):
            return None

        # let default json encoder resolves the rest of the issues
        return obj

    def __setitem__(self, key, value):
        key = JsonDict.curate(key)
        value = JsonDict.curate(value)
        super().__setitem__(key, value)


class FeatureStore:

    """
    A wrapper of data structure to store the features. Use dict to describe the feature name.
    For each object, define and limit the feature names it supported.
    It can define the initial value. By default it is NaN, therefore it provides convenience to cover some of
    the edge cases where the graph features cannot be calculated due to insufficient node numbers.
    (e.g., only two nodes --> cannot perform triangulation)
    """

    @staticmethod
    def __init_dict_value(which_dict, keys: List[str], init_value=np.nan):
        for k in keys:
            which_dict[k] = init_value

    @property
    def _feature_names(self):
        return self.__feature_names

    @property
    def size(self):
        return len(self.__feature)

    @property
    def feature_dict(self):
        return deepcopy(self.__feature)

    def __init__(self, feature_names: List[str], init_value=np.nan):
        self.__feature: Dict = dict()
        assert len(feature_names) > 0
        self.__feature_names = feature_names
        self.__init_value = init_value
        FeatureStore.__init_dict_value(self.__feature, feature_names, init_value=init_value)

    def __setitem__(self, key, value):
        assert key in self.__feature_names
        self.__feature[key] = value

    def __getitem__(self, item):
        assert item in self.__feature_names
        return self.__feature[item]
