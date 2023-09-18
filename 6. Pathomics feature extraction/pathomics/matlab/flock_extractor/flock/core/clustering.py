from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
from sklearn.cluster import MeanShift, KMeans


class AbstractClusterFuncWrapper(ABC):
    """
    For now, a simple wrapper for sklearn's mean-shift clustering module, since the the corresponding
    base type of sklearn may not be exposed.
    Also, easier to change if other type of clustering scheme is required.
    This unifies the interface of required methods for the clustering algorithm, and improves the extensibility
    when another third-party clustering algorithm outside of sklearn is required, and if its interface is not
    identical to sklearn's clustering algorithms (e.g. HDBscan, and it has no clusters_centers_ attribute)
    Provides "fit", "fit_predict", "predict", and "get_oarams"
    """

    @property
    def clusters_centers_(self):
        """
        Center of clusters. Override if necessary. For instance for certain cluster algorithm the center is meaningless
        and therefore not provided.
        Returns:

        """
        return self.clustering.cluster_centers_

    def __init__(self, clustering):
        self.clustering = clustering

    def fit(self, x: np.ndarray) -> 'AbstractClusterFuncWrapper':
        """
        Fluent interface of fit.
        Args:
            x:

        Returns:

        """
        self.clustering.fit(x)
        return self

    def fit_predict(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit the model and predict the cluster label of the input.
        Args:
            x: input data to fit the model
            y: data to predict

        Returns:
            np.ndarray: An array of cluster label, wherein the output[i] is the label of y[i]
        """
        return self.clustering.fit_predict(x, y)

    def predict(self, y: np.ndarray) -> np.ndarray:
        """
        Predict the cluster label of the input.
        Args:
            y: data to predict

        Returns:
            np.ndarray: An array of cluster label, wherein the output[i] is the label of y[i]
        """
        return self.clustering.predict(y)

    def get_params(self) -> Dict:
        """
        Get the parameters and hyperparameters of the clustering methods.
        Returns:

        """
        return self.clustering.get_params()

    @classmethod
    @abstractmethod
    def build(cls, *args, **kwargs):
        """
        Factory method.
        Args:
            *args:
            **kwargs:

        Returns:

        """
        raise NotImplementedError


class MeanShiftWrapper(AbstractClusterFuncWrapper):
    """
    A wrapper of sklearn's MeanShift algorithm.
    """

    def __init__(self, clustering: MeanShift):
        super().__init__(clustering)

    @classmethod
    def build(cls,
              bandwidth: float = None,
              seeds: np.ndarray = None,
              bin_seeding: bool = False,
              min_bin_freq: int = 1,
              cluster_all: bool = True,
              n_jobs: int = None,
              max_iter: int = 300):

        mean_shift = MeanShift(bandwidth=bandwidth, seeds=seeds, bin_seeding=bin_seeding, min_bin_freq=min_bin_freq,
                               cluster_all=cluster_all, n_jobs=n_jobs, max_iter=max_iter)
        return cls(mean_shift)


class KMeansWrapper(AbstractClusterFuncWrapper):
    """
    A wrapper of kmean's algorithm
    """
    def __init__(self, clustering: KMeans):
        super().__init__(clustering)

    @classmethod
    def build(cls, n_clusters, *args, **kwargs):
        k_means = KMeans(*args, n_clusters=n_clusters, **kwargs)
        return cls(k_means)
