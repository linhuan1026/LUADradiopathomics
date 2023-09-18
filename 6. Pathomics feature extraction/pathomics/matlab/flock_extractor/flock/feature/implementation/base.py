from abc import ABC, abstractmethod
from typing import Dict, List
from pathomics.matlab.flock_extractor.flock.feature.util import FeatureStore
import numpy as np


class FeatureSkeleton(ABC):
    """
    Skeleton for all Features.
    """
    __feature_store: FeatureStore

    @abstractmethod
    def features(self) -> Dict:
        raise NotImplementedError


class NamedFeatureSkeleton(FeatureSkeleton):
    """
    Skeleton for all Features that need predefined default values.
    This is used for features that may throw exception on edge cases. (use nan as default values)
    """
    __feature_store: FeatureStore

    def feature_placeholder(self, feature_names: List[str], init_value):
        self.__feature_store = FeatureStore(feature_names, init_value)

    def _feature_set_value(self, feature_name_single: str, value):
        self.__feature_store[feature_name_single] = value

    @property
    def feature_store(self):
        return self.__feature_store

    @property
    @abstractmethod
    def feature_names(self):
        raise NotImplementedError

    def define_feature(self, init_value=np.asarray([np.nan])):
        self.feature_placeholder(self.feature_names, init_value)

    @abstractmethod
    def init_value_curate(self):
        raise NotImplementedError

    def __init__(self, init_value=np.asarray([np.nan])):
        self.define_feature(init_value)
        self.init_value_curate()
