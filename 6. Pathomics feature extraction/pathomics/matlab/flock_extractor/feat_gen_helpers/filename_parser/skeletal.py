from abc import ABC, abstractmethod
import os
from typing import List
from lazy_property import LazyProperty


class AbstractParser(ABC):
    """
    Base class of filename parser to extract patch information from the filename.
    Assume there is a delimiter separates different components
    """
    __components: List[str]

    @property
    @abstractmethod
    def patient_idx(self):
        """
        Patient idx or slide idx.
        Returns:

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def coordinates(self):
        raise NotImplementedError

    @property
    def filename(self):
        return self.__filename

    @staticmethod
    def validate_delimiter(delimiter):
        """
        Make sure the delimiter is not None --> if None set to empty string
        Args:
            delimiter:

        Returns:

        """
        if delimiter is None:
            delimiter = ''
        return delimiter

    def __init__(self, filename, delimiter):
        self.__filename = filename
        self.__delimiter = AbstractParser.validate_delimiter(delimiter)

    @staticmethod
    def filename_split_helper(filename: str, delimiter: str = '_'):
        """
        A simple wrapper function to split filenames by the delimiter
        Args:
            filename:
            delimiter:

        Returns:

        """
        basename = os.path.basename(filename)
        fbase, _ = os.path.splitext(basename)
        components = fbase.split(delimiter)
        return components

    @LazyProperty
    def components(self):
        """
        A lazy evaluated property of components separated by the delimiter
        Returns:

        """
        self.__components = AbstractParser.filename_split_helper(self.filename, self.delimiter)
        assert len(self.__components) == 8
        return self.__components

    @property
    def delimiter(self):
        return self.__delimiter

    @abstractmethod
    def reconstruct_from_coords(self, coord):
        """
        With the given patient/slide idx, reconstruct the filename from the coordinates of other patches
        belonging to the same slide/patient
        Args:
            coord:

        Returns:

        """
        raise NotImplementedError()
