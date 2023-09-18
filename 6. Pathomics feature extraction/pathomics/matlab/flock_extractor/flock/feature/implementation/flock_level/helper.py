import numpy as np
from scipy.stats import kurtosis, skew
from typing import Dict, Sequence, Union, Callable, Set

eps = np.finfo(float).eps


def _range(src_data: np.ndarray):
    """
    Helper function to calculate the range of the given array (min - max)
    Args:
        src_data:

    Returns:

    """
    max_val = np.max(src_data)
    min_val = np.min(src_data)
    return max_val - min_val


class FeatReduction:
    """
    Reduce the feature vector to a scalar by given set of statistics instead of hard coding.
    """

    KEY_MIN: str = 'min'
    KEY_MAX: str = 'max'
    KEY_RANGE: str = 'range'
    KEY_MEAN: str = 'mean'
    KEY_MEDIAN: str = 'median'
    KEY_STD: str = 'std'
    KEY_KURTOSIS: str = 'kurtosis'
    KEY_SKEWNESS: str = 'skewness'

    KEY_SUM: str = 'sum'

    name2func: Dict[str, Callable[[np.ndarray], Union[float, np.ndarray]]] = {
        KEY_MIN: np.min,
        KEY_MAX: np.max,
        KEY_RANGE: _range,
        KEY_MEAN: np.mean,
        KEY_MEDIAN: np.median,
        KEY_STD: np.std,
        KEY_KURTOSIS: kurtosis,
        KEY_SKEWNESS: skew,

        KEY_SUM: np.sum,
    }

    CONST_STAT: Sequence[str] = [
        KEY_MIN,
        KEY_MAX,
        KEY_RANGE,
        KEY_MEAN,
        KEY_MEDIAN,
        KEY_STD,
        KEY_KURTOSIS,
        KEY_SKEWNESS,
    ]

    @staticmethod
    def to_prefix(pref: str) -> str:
        """
        Prefix of feature names. The final feature name is [prefix][stat_name]. E.g., area_mean
        Args:
            pref:

        Returns:

        """
        if pref is None:
            return ''
        return str(pref)

    @staticmethod
    def to_np_data(src_data) -> np.ndarray:
        """
        Convert input into numpy array. If empty (e.g. no nodes --> nan), return nan.
        """
        result = np.asarray(src_data)
        if result.shape[0] == 0:
            result = np.asarray([np.nan])
        return result

    @staticmethod
    def dict_assign(result, name_prefix, stat_name, value):
        """
        Helper function to assign the feature with its name into a dict
        Args:
            result:
            name_prefix:
            stat_name:
            value:

        Returns:

        """
        result[name_prefix + stat_name] = value

    @staticmethod
    def statisti_reduce_single(src_data: Union[Sequence[float], np.ndarray], name_prefix: str,
                               stat_name: str) -> Dict[str, Union[float, np.ndarray]]:
        """

        Args:
            src_data: source data. Will be converted into np.ndarray
            name_prefix: prefix of feature name. The final feature name is [prefix][stat_name]. E.g., area_mean
            stat_name: Name of the statistics, e.g., mean/std/...

        Returns:
            Dict: [prefix][stat_name] -> stat result
        """
        func:  Callable[[np.ndarray], Union[float, np.ndarray]] = FeatReduction.name2func[stat_name]
        single_entry = dict()
        outcome = func(src_data)
        FeatReduction.dict_assign(single_entry, name_prefix, stat_name, outcome)
        return single_entry

    @staticmethod
    def statisti_reduce_helper_vector(src_data: Union[Sequence[float], np.ndarray],
                                      name_prefix: str,
                                      stat_name_list: Sequence[str]) -> Dict[str, Union[float, np.ndarray]]:
        """
        Vectorization of statisti_reduce_single. Calculate each statistics by the given stat_name_list
        Args:
            src_data:
            name_prefix:
            stat_name_list:

        Returns:

        """
        result = dict()
        for stat_name in stat_name_list:
            single_entry = FeatReduction.statisti_reduce_single(src_data, name_prefix, stat_name)
            result.update(single_entry)
        return result

    @staticmethod
    def statisti_reduce(src_data: Union[Sequence[float], np.ndarray], name_prefix: str):
        """
        Wrapper of statisti_reduce_helper_vector. stat_name_list is given by the CONST_STAT
        Args:
            src_data:
            name_prefix:

        Returns:

        """
        name_prefix = FeatReduction.to_prefix(name_prefix)
        src_data = FeatReduction.to_np_data(src_data)
        return FeatReduction.statisti_reduce_helper_vector(src_data, name_prefix, FeatReduction.CONST_STAT)

    # @staticmethod
    # def statisti_reduce(src_data: Union[Sequence[float], np.ndarray], name_prefix: str):
    #     name_prefix = FeatReduction.to_prefix(name_prefix)
    #     src_data = FeatReduction.to_np_data(src_data)
    #     result: Dict[str, float] = dict()
    #
    #     min_data = np.min(src_data)
    #     FeatReduction.dict_assign(result, name_prefix, FeatReduction.KEY_MIN, min_data)
    #     max_data = np.max(src_data)
    #     FeatReduction.dict_assign(result, name_prefix, FeatReduction.KEY_MAX, max_data)
    #
    #     FeatReduction.dict_assign(result, name_prefix, FeatReduction.KEY_RANGE, max_data - min_data)
    #
    #     FeatReduction.dict_assign(result, name_prefix, FeatReduction.KEY_MEAN, np.mean(src_data))
    #     FeatReduction.dict_assign(result, name_prefix, FeatReduction.KEY_STD, np.std(src_data))
    #     FeatReduction.dict_assign(result, name_prefix, FeatReduction.KEY_MEDIAN, np.median(src_data))
    #
    #     FeatReduction.dict_assign(result, name_prefix, FeatReduction.KEY_KURTOSIS, kurtosis(src_data, axis=None))
    #     FeatReduction.dict_assign(result, name_prefix, FeatReduction.KEY_SKEWNESS, skew(src_data, axis=None))
    #     return result

    @staticmethod
    def flattened_ratio(ratio_dict: Dict[int, Union[Sequence[float], np.ndarray]], positive_only: bool) -> np.ndarray:
        """
        flatten the dict of intersection area ratio (intersection area divided by the polygon area) into a vector
        Args:
            ratio_dict: polygon idx --> intersection
            positive_only:

        Returns:

        """
        to_concat = [np.asarray(x) for x in ratio_dict.values()]
        if len(to_concat) == 0:
            flattened = np.empty(0)
        else:
            flattened = np.concatenate([np.asarray(x) for x in ratio_dict.values()])
        flattened = FeatReduction.to_np_data(flattened)
        if not positive_only:
            return flattened
        return flattened[flattened > 0]

    @staticmethod
    def ratio_reduce(ratio_dict: Dict[int, Union[Sequence[float], np.ndarray]],
                     thresholds: Union[Sequence[float], np.ndarray],
                     name_pref: str,
                     denominator: float):
        """
        Caluclate the count of intersection ratio that is above the threshold in all given thresholds.

        Ratio dict is the intersection area ratio over the source or target polygon area of a pair of
        intersected polygons. Thresholds determines whether the intersection area is small/median/large.
        denominator simply defines whether the output should be normalized/rescaled

        Args:
            ratio_dict: the intersection area ratio over the source or target polygon area of a pair of
                intersected polygons.
            thresholds: Thresholds determines whether the intersection area is small/median/large.
            name_pref: prefix of the feature name
            denominator: whether rescale the output. If 1 --> returns the count. If # of cluster, return the fraction.

        Returns:

        """
        # ratio_dict: polygon_idx --> list of intersection ratio
        assert denominator != 0
        ratio_arr: np.ndarray = FeatReduction.flattened_ratio(ratio_dict, positive_only=True)
        ratio_arr = FeatReduction.to_np_data(ratio_arr)
        name_pref = FeatReduction.to_prefix(name_pref)
        result: Dict = dict()
        for thresh in thresholds:
            stat_name = f"{FeatReduction.KEY_SUM}_{thresh}"
            stat_result = sum(ratio_arr > thresh) / denominator
            FeatReduction.dict_assign(result, name_pref, stat_name, stat_result)
        return result

    @staticmethod
    def phenotype_intersect_reduce(intersect_type_count: Union[Sequence[float], np.ndarray],
                                   cluster_of_type_count: int,
                                   name_pref: str,
                                   phenotype_id: int):
        """
        Helper function to reduce inter/intra phenotype features. Calculate sum and the fraction
         (sum divided by # of clusters of the given phenotype)
         Input is an array of booleans marks whether each pair satisfies the condition of inter or intra-phenotype
         statistics (e.g. whether the pair has the same type to the given phenotype, or both have different types)
        Args:
            intersect_type_count:
            cluster_of_type_count:
            name_pref:
            phenotype_id:

        Returns:

        """
        new_name_pref = f"{name_pref}{phenotype_id}_"
        result = FeatReduction.statisti_reduce_single(intersect_type_count, new_name_pref,
                                                      FeatReduction.KEY_SUM)
        new_name_pref_ratio = f"{name_pref}{phenotype_id}_ratio"
        intersect_type_count_frac = intersect_type_count.astype(np.float32) / cluster_of_type_count
        result_ratio = FeatReduction.statisti_reduce_single(intersect_type_count_frac, new_name_pref_ratio,
                                                            FeatReduction.KEY_SUM)
        result.update(result_ratio)
        return result

