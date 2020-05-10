"""
M4 Dataset
"""
from collections import OrderedDict
from typing import Dict

import numpy as np
from gluonts.dataset.common import ListDataset
from gluonts.dataset.repository.datasets import get_dataset

from datasets.dataset import Dataset


class M4Dataset(Dataset):
    """
    M4 Dataset
    """

    def __init__(self):
        self.frequency_map = {
            'yearly': '12M',
            'quarterly': '3M',
            'monthly': 'M',
            'weekly': '7D',
            'daily': 'D',
            'hourly': 'H'
        }

        self.horizon_map = {
            'yearly': 6,
            'quarterly': 8,
            'monthly': 18,
            'weekly': 13,
            'daily': 14,
            'hourly': 48
        }

    def load(self, frequency: str, subset_filter: str, training: bool) -> ListDataset:
        """
        Load M4 subset
        :param frequency:
        :param subset_filter:
        :param training:
        :return:
        """
        dataset = get_dataset(f'm4_{subset_filter}')
        return ListDataset(list(dataset.train if training else dataset.test), freq=frequency)

    def evaluate_subset(self, forecast: np.ndarray, subset_filter: str) -> np.ndarray:
        target_dataset = self.load(frequency=self.frequency_map[subset_filter],
                                   subset_filter=subset_filter,
                                   training=False)
        horizon = self.horizon_map[subset_filter]
        target = np.array([x['target'][-horizon:] for x in target_dataset])
        return 200 * np.abs(forecast - target) / (np.abs(target) + np.abs(forecast))

    def evaluate(self, forecast: np.ndarray) -> Dict[str, float]:
        scores = [
            self.evaluate_subset(forecast[:23000, :6], subset_filter='yearly'),
            self.evaluate_subset(forecast[23000:47000, :8], subset_filter='quarterly'),
            self.evaluate_subset(forecast[47000:95000, :18], subset_filter='monthly'),
            self.evaluate_subset(forecast[95000:95359, :13], subset_filter='weekly'),
            self.evaluate_subset(forecast[95359:99586, :14], subset_filter='daily'),
            self.evaluate_subset(forecast[99586:100000, :48], subset_filter='hourly')
        ]
        return self.summarize_groups({
            'yearly': np.mean(scores[0]),
            'quarterly': np.mean(scores[1]),
            'monthly': np.mean(scores[2]),
            'weekly': np.mean(scores[3]),
            'daily': np.mean(scores[4]),
            'hourly': np.mean(scores[5])
        })

    def summarize_groups(self, scores) -> Dict[str, float]:
        """
        Re-group scores respecting M4 rules.
        :param scores: Scores per group.
        :return: Grouped scores.
        """
        scores_summary = OrderedDict()

        def subset_count(subset_filter):
            return len(self.load(frequency=self.frequency_map[subset_filter],
                                 subset_filter=subset_filter,
                                 training=False))

        weighted_score = {}
        for g in ['yearly', 'quarterly', 'monthly']:
            weighted_score[g] = scores[g] * subset_count(g)
            scores_summary[g] = scores[g]

        others_score = 0
        others_count = 0
        for g in ['weekly', 'daily', 'hourly']:
            others_score += scores[g] * subset_count(g)
            others_count += subset_count(g)
        weighted_score['others'] = others_score
        scores_summary['others'] = others_score / others_count

        average = np.sum(list(weighted_score.values())) / 100000
        scores_summary['average'] = average

        return scores_summary
