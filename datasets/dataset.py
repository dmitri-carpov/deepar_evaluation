"""
Dataset abstraction.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict

import numpy as np
from gluonts.dataset.common import ListDataset

class Dataset(ABC):
    """
    Abstract dataset class
    """
    @abstractmethod
    def load(self, frequency: str, subset_filter: str, training: bool) -> ListDataset:
        """
        Load subset of the dataset.

        :param frequency: Dataset frequency
        :param subset_filter: Filter for dataset
        :param training: Training split if True test otherwise
        :return: GluonTS ListDataset
        """

    @abstractmethod
    def evaluate(self, forecast: np.ndarray) -> Dict[str, float]:
        """
        Evaluate forecasts.

        :param forecast: Forecast to evaluate
        If None then the forecast for the whole dataset is expected.
        :return: Scores for each subset and average.
        """

    @staticmethod
    def to_array(dataset: ListDataset) -> np.ndarray:
        return np.array([v['target'] for v in dataset])
