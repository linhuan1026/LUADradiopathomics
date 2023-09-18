from typing import List, Iterable, Tuple, Callable, Dict, Union, Hashable
import pandas as pd
import numpy as np
import multiprocessing as mp
from sklearn.neighbors import KDTree
from feat_gen_helpers.filename_parser import GoyaParser, AbstractParser
import os


def feats_from_sheet(feat_sheet: pd.DataFrame, attr_names: List[str]):
    """
    Select feature from the pandas.DataFrame by the given column name
    Args:
        feat_sheet:
        attr_names: name of features (columns)

    Returns:

    """
    feat_list = []
    for attr_name_single in attr_names:
        curr_feat = np.asarray(feat_sheet[attr_name_single])[:, None]
        feat_list.append(curr_feat)
    feat_out = np.hstack(feat_list)
    return feat_out


def l2_norm(x, y):
    """Calculate l2 norm (distance) of `x` and `y`.
    Args:
        x (numpy.ndarray or cupy): (batch_size, num_point, coord_dim)
        y (numpy.ndarray): (batch_size, num_point, coord_dim)
    Returns (numpy.ndarray): (batch_size, num_point,)
    """
    return ((x - y) ** 2).sum(axis=2)


def farthest_point_sampling(pts, k, initial_idx=None, metrics=l2_norm,
                            skip_initial=False):
    """farthest point sampling
    Code referenced from below link by @Graipher
    https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python
    Args:
        pts (numpy.ndarray or cupy.ndarray): 2-dim array (num_point, coord_dim)
            or 3-dim array (batch_size, num_point, coord_dim)
            When input is 2-dim array, it is treated as 3-dim array with
            `batch_size=1`.
        k (int): number of points to sample
        initial_idx (int): initial index to start farthest point sampling.
            `None` indicates to sample from random index,
            in this case the returned value is not deterministic.
        metrics (callable): metrics function, indicates how to calc distance.
        skip_initial (bool): If True, initial point is skipped to store as
            farthest point. It stabilizes the function output.
    Returns (tuple): `indices` and `distances`.
        indices (numpy.ndarray or cupy.ndarray): 2-dim array (batch_size, k, )
            indices of sampled farthest points.
            `pts[indices[i, j]]` represents `i-th` batch element of `j-th`
            farthest point.
        distances (numpy.ndarray or cupy.ndarray): 3-dim array
            (batch_size, k, num_point)
    """

    if pts.ndim == 2:
        # insert batch_size axis
        pts = pts[None, ...]
    assert pts.ndim == 3
    batch_size, num_point, coord_dim = pts.shape
    indices = np.zeros((batch_size, k,), dtype=np.int64)

    # distances[bs, i, j] is distance between i-th farthest point `pts[bs, i]`
    # and j-th input point `pts[bs, j]`.
    distances = np.zeros((batch_size, k, num_point), dtype=np.float64)
    if initial_idx is None:
        indices[:, 0] = np.random.randint(len(pts))
    else:
        indices[:, 0] = initial_idx

    batch_indices = np.arange(batch_size)
    farthest_point = pts[batch_indices, indices[:, 0]]
    # minimum distances to the sampled farthest point

    min_distances = metrics(farthest_point[:, None, :], pts)
    if skip_initial:
        # Override 0-th `indices` by the farthest point of `initial_idx`
        indices[:, 0] = np.argmax(min_distances, axis=1)
        farthest_point = pts[batch_indices, indices[:, 0]]
        min_distances = metrics(farthest_point[:, None, :], pts)

    distances[:, 0, :] = min_distances
    for i in range(1, k):
        indices[:, i] = np.argmax(min_distances, axis=1)
        farthest_point = pts[batch_indices, indices[:, i]]
        dist = metrics(farthest_point[:, None, :], pts)
        distances[:, i, :] = dist
        min_distances = np.minimum(min_distances, dist)
    return indices, distances


def naive_mp_map_helper(samples: Iterable, callback: Callable, n_workers: int):
    """
    simple multiprocessing wrapper of the pool.map method.
    If n_workers is zero, then simply use a list comprehensive loop
    Args:
        samples:
        callback:
        n_workers:

    Returns:

    """
    if n_workers == 0:
        output = [callback(x) for x in samples]
    else:
        pool = mp.Pool(n_workers)
        output = pool.map(callback, samples)
        pool.close()
        pool.join()
    return output


class ImageRecord:
    """
    Wrapper class for the record of patch information -- coordinate and other information in the name
    It is basically associated with a Dict[str, Dict[str, List[Union[np.ndarray, AbstractParser]]]]
    Image Idx --> Inner Dict <KEY_COORDS? --> list of coordinates. <KEY_PARSER> --> list of parser
    """
    KEY_COORDS: str = 'coords'
    KEY_PARSER: str = 'parser'

    __records: Dict[str, Dict[str, List[Union[np.ndarray, AbstractParser]]]]

    @staticmethod
    def idx_coord_helper(sample_in: AbstractParser) -> Tuple[str, np.ndarray, AbstractParser]:
        """
        helper function for the multiprocessing
        Args:
            sample_in:

        Returns:

        """
        return sample_in.patient_idx, sample_in.coordinates, sample_in

    @staticmethod
    def idx_coord(parsers: List[GoyaParser], n_workers: int) -> List[Tuple[str, np.ndarray, AbstractParser]]:
        """
        Get the image idx, coordiantes, and the parser itself from the parsers. Support multiprocessing
        Args:
            parsers:
            n_workers:

        Returns:

        """
        return naive_mp_map_helper(parsers, ImageRecord.idx_coord_helper, n_workers)

    @staticmethod
    def idx_coord_aggregate(idx_coord_pairs: List[Tuple[str, np.ndarray, AbstractParser]],
                            n_workers: int = 0) -> Dict[str, Dict[str, List[Union[np.ndarray, AbstractParser]]]]:
        """
        Aggregate list output of naive_mp_map_helper into the descired
        Dict[str, Dict[str, List[Union[np.ndarray, AbstractParser]]]].
        Image Idx --> Inner Dict[ key_name --> corresponding list of values]
        Args:
            idx_coord_pairs:
            n_workers:

        Returns:

        """
        n_workers: int  # due to that list accumulation may not be atomic -- in most of the
        # cases it may not worth the parallelized implementation.
        # leave it here for future adaptation

        result: Dict[str, Dict[str, List[Union[np.ndarray, AbstractParser]]]] = dict()
        for pair in idx_coord_pairs:
            image_idx, coord, parser = pair
            result[image_idx] = result.get(image_idx, dict())
            result[image_idx][ImageRecord.KEY_COORDS] = result[image_idx].get(ImageRecord.KEY_COORDS, [])
            result[image_idx][ImageRecord.KEY_COORDS].append(coord)
            # only need to write once.
            result[image_idx][ImageRecord.KEY_PARSER] = result[image_idx].get(ImageRecord.KEY_PARSER, [])
            result[image_idx][ImageRecord.KEY_PARSER].append(parser)
        return result

    # def coords_by_image_idx(self, image_idx):
    #     return self.__records[image_idx][ImageRecord.KEY_COORDS]
    #
    # def parser_by_image_idx(self, image_idx):
    #     return self.__records[image_idx][ImageRecord.KEY_PARSER]

    @property
    def records(self) -> Dict:
        return self.__records

    def __init__(self, coords_by_image: Dict[str, Dict[str, List[Union[np.ndarray, AbstractParser]]]]):
        self.__records = coords_by_image

    @classmethod
    def build(cls, parsers, n_workers):
        idx_coord_pair_list: List[Tuple[str, np.ndarray, AbstractParser]] = ImageRecord.idx_coord(parsers, n_workers)
        coords_by_image = ImageRecord.idx_coord_aggregate(idx_coord_pair_list)
        return cls(coords_by_image)


class CoordsAggregator:
    """
    Use farthest point sampling to sample a fixed number of patches by their top-left coordinates.
    Then query all neighboring patches of the right/bottom direction within a given range.
    This is a workaround to sample and group small patches into larger tiles.
    """

    @staticmethod
    def num_of_step_per_axis(step_size, tile_size):
        """
        Since the coordinates might be given by a step size rather than the pixel size directly --> define the
        conversion here.
        For Goya --> step = 256, actual tile_size converted into step size scales are tile // 256
        Args:
            step_size:
            tile_size:

        Returns:

        """
        return tile_size // step_size - 1

    @staticmethod
    def coord_query_kd_tree(all_pts: np.ndarray, num_to_sample, step_size, tile_size) -> Tuple[np.ndarray, np.ndarray]:
        """
        Don't find a range tree implementation. So I simply use a KD-tree as a workaround.
        The KD-tree first query all points within 2 * converted tile size, and then select the patches
        within the target tile.
        Args:
            all_pts:
            num_to_sample:
            step_size:
            tile_size:

        Returns:

        """
        # coords_list: List[np.ndarray]
        # all_pts = np.vstack(coords_list)
        # manhattan distance boundary (width + height).
        # since coords are calculated by stride nums --> tile_size // step_size as the distance unit
        # times two -- include both the width and height (horizontal and vertical)
        # minus one as the top left is already included
        distance_thresh = 2 * CoordsAggregator.num_of_step_per_axis(step_size, tile_size)
        indices, distances = farthest_point_sampling(all_pts, num_to_sample)
        sampled_pts = np.atleast_2d(all_pts[indices.squeeze()])

        # for each of the sampled pts --> find range
        query_all = KDTree(all_pts, metric='manhattan')
        query_results_idx = query_all.query_radius(sampled_pts, r=distance_thresh)
        return sampled_pts, query_results_idx

    @staticmethod
    def coord_query_bbox(coords_in: np.ndarray, top_left: np.ndarray, tile_size: int, step_size: int):
        """
        Further select the kd-tree queried points within the bounding box of the tile.
        Args:
            coords_in:
            top_left:
            tile_size:
            step_size:

        Returns:

        """
        box_size = CoordsAggregator.num_of_step_per_axis(step_size, tile_size)
        c1_low, c2_low = top_left
        c1_high, c2_high = top_left + box_size
        coords = coords_in  # just for simplicity for renaming
        coords = coords[coords[:, 0] >= c1_low]
        coords = coords[coords[:, 0] <= c1_high]

        coords = coords[coords[:, 1] >= c2_low]
        coords = coords[coords[:, 1] <= c2_high]
        return coords

    @staticmethod
    def coord_query_results_bbox_helper(all_pts: np.ndarray, sampled_pts: np.ndarray, query_results_idx: np.ndarray,
                                        tile_size: int, step_size: int) -> Dict[int, np.ndarray]:
        """
        Aggregate the output of coord_query_bbox into a dict
        Args:
            all_pts:
            sampled_pts:
            query_results_idx:
            tile_size:
            step_size:

        Returns:
            Dict: tile id -> an array of selected coordinates
        """
        assert sampled_pts.shape[0] == query_results_idx.shape[0]
        tile_id_to_coords = dict()
        for idx, (top_left, query_indices) in enumerate(zip(sampled_pts, query_results_idx)):
            coords_input = all_pts[query_indices]
            coords_in_bounding_box = CoordsAggregator.coord_query_bbox(coords_input, top_left, tile_size, step_size)
            tile_id_to_coords[int(idx)] = coords_in_bounding_box
        return tile_id_to_coords

    @staticmethod
    def coord_query_helper_wrapper(image_idx: str,
                                   coords_list: List[np.ndarray],
                                   num_to_sample: int,
                                   step_size: int,
                                   tile_size: int) -> Tuple[str, Dict]:
        """
        For each image_idx, get the dict of tile_id -> array of selected coordinates
        Args:
            image_idx:
            coords_list:
            num_to_sample:
            step_size:
            tile_size:

        Returns:
            Tuple: str: image_idx, Dict: tile id -> an array of selected coordinates
        """
        all_pts = np.vstack(coords_list)
        # return ndarray of ndarray
        sampled_pts, query_results_idx = CoordsAggregator.coord_query_kd_tree(all_pts, num_to_sample,
                                                                              step_size, tile_size)
        tile_id_to_coords = CoordsAggregator.coord_query_results_bbox_helper(all_pts, sampled_pts,
                                                                             query_results_idx,
                                                                             tile_size, step_size)
        return image_idx, tile_id_to_coords

    def coord_query_single(self):
        """
        Without multiprocessing
        Returns:

        """
        result = dict()
        for image_idx, record_dict in self.__coords_by_image.items():
            coords_list = record_dict[ImageRecord.KEY_COORDS]
            _, tile_id_to_coords = CoordsAggregator.coord_query_helper_wrapper(image_idx,
                                                                               coords_list,
                                                                               self.num_to_sample.value,
                                                                               self.step_size.value,
                                                                               self.tile_size.value)
            result[image_idx] = tile_id_to_coords
        return result

    def coord_query(self):
        coord_results = self.coord_query_single()
        return CoordsAggregator.filter_min_patch_num(coord_results, self._min_patch_num)

    @staticmethod
    def _dict_purge(which_dict: Dict, keys_purge_list: List[Hashable]):
        """
        Remove keys from dict
        Args:
            which_dict:
            keys_purge_list:

        Returns:

        """
        for idx_to_remove in keys_purge_list:
            which_dict.pop(idx_to_remove)
        return which_dict

    @staticmethod
    def record_curate(record, min_num_patch):
        """
        Remove tiles without sufficient number of patches given by the threshold
        Args:
            record:
            min_num_patch:

        Returns:

        """
        tile_to_remove = []
        for tile_id, coords in record.items():
            if coords.shape[0] >= min_num_patch:
                continue
            tile_to_remove.append(tile_id)
        return CoordsAggregator._dict_purge(record, tile_to_remove)

    @staticmethod
    def filter_min_patch_num(coord_results, min_num_patch: int = 15):
        # hate this
        image_to_remove = []
        for image_idx, record in coord_results.items():
            record = CoordsAggregator.record_curate(record, min_num_patch)
            if len(record) > 0:
                continue
            image_to_remove.append(image_idx)
        coord_results = CoordsAggregator._dict_purge(coord_results, image_to_remove)
        return coord_results

    # todo - for now there is no need to parallelize this operation
    # def __closure_helper(self, record: Iterable):
    #     image_idx, record_dict = record
    #     coords_list = record_dict[ImageRecord.KEY_COORDS]
    #     image_idx, tile_id_to_coords = CoordsAggregator.coord_query_helper_wrapper(image_idx,
    #                                                                                coords_list,
    #                                                                                self.num_to_sample.value,
    #                                                                                self.step_size.value,
    #                                                                                self.tile_size.value)
    #     return image_idx, tile_id_to_coords
    #
    # def coord_query_parallel(self):
    #     coord_img_item = self.__coords_by_image.items()
    #     pool = mp.Pool(self._num_workers)
    #     output_id_to_coords: List[Tuple] = pool.map(self.__closure_helper, coord_img_item)
    #     output_dict = dict(output_id_to_coords)
    #     return output_dict

    def __init__(self, coords_by_image: Dict[str, Dict[str, List[Union[np.ndarray, AbstractParser]]]],
                 num_to_sample: int, step_size: int, tile_size: int, num_workers: int = 0,
                 min_patch_num: int = 4):
        """

        Args:
            coords_by_image: from ImageRecord objects
            num_to_sample:
            step_size: the step size that convert the pixel size to what used in coordinates of the parser/filename
            tile_size: tile size in pixels
            num_workers:
            min_patch_num:
        """
        self.__coords_by_image = coords_by_image
        self.num_to_sample: mp.Value = mp.Value('i', num_to_sample)
        self.step_size = mp.Value('i', step_size)
        self.tile_size = mp.Value('i', tile_size)
        self._num_workers = num_workers
        self._min_patch_num = min_patch_num


def retrieve_filename(image_record: ImageRecord, image_tiles_coords: Dict,
                      root_dir: str = '', ext_with_dot: str = ''):
    """
    Get all sheet filenames of corresponding patches of tiles
    Args:
        image_record:
        image_tiles_coords:
        root_dir:
        ext_with_dot:

    Returns:
        Dict: image_idx --> Dict[tile_id -> list of sheet names]
    """
    image_record_dict = image_record.records
    image_level_dict = dict()
    for image_idx, tile_coords in image_tiles_coords.items():
        parser: AbstractParser = image_record_dict[image_idx][ImageRecord.KEY_PARSER][0]
        tid_to_files_basename = {tile_id: parser.reconstruct_from_coords(coord_in_tile)
                                 for tile_id, coord_in_tile in tile_coords.items()}
        tid_to_fullname = {int(tile_id): [os.path.join(root_dir, f"{f}{ext_with_dot}") for f in filename]
                           for tile_id, filename in tid_to_files_basename.items()}

        image_level_dict[str(image_idx)] = tid_to_fullname
    return image_level_dict


def merge_csv(filename_list: List[str]):
    """
    Read all csv files by the given filename list, and merge them into a single pandas.DataFrame
    Args:
        filename_list: filenames

    Returns:

    """
    sheet_list = []
    for fname in filename_list:
        if not os.path.exists(fname):
            continue
        sheet = pd.read_csv(fname)
        sheet_list.append(sheet)
    assert len(sheet_list) > 0
    return pd.concat(sheet_list)
