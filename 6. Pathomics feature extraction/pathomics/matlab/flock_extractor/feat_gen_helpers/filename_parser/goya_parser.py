from .skeletal import AbstractParser
from typing import List
import numpy as np
from lazy_property import LazyProperty


class GoyaParser(AbstractParser):
    """
    For goya dataset
    """
    IDX_PID: int = 5
    IDX_COORD1: int = 6
    IDX_COORD2: int = 7

    __components: List[str]

    def __init__(self, filename: str, delimiter: str = '_'):
        super().__init__(filename, delimiter)

    @property
    def patient_idx(self) -> str:
        return self.components[GoyaParser.IDX_PID]

    @property
    def coordinates(self):
        c1 = int(self.components[GoyaParser.IDX_COORD1])
        c2 = int(self.components[GoyaParser.IDX_COORD2])
        return np.asarray([c1, c2])

    def reconstruct_from_coord_1d(self, coord) -> str:
        assert coord.ndim == 1
        prefix = self.components[0: -2]
        coord_str_suffix = [str(x) for x in coord]
        component_reconst: List[str] = prefix + coord_str_suffix
        return self.delimiter.join(component_reconst)

    def reconstruct_from_coords(self, coords: np.ndarray) -> List[str]:
        coords = np.atleast_2d(coords)
        return [self.reconstruct_from_coord_1d(c) for c in coords]

