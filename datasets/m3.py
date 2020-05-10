"""
M3 Dataset
"""

import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import requests
from gluonts.dataset.common import ListDataset

from config import DATASETS_PATH
from datasets.dataset import Dataset


class M3Dataset(Dataset):
    def __init__(self):
        self.dataset_path = os.path.join(DATASETS_PATH, 'M3')
        self.dataset_file = os.path.join(self.dataset_path, 'M3C.xls')
        if not os.path.isfile(self.dataset_file):
            Path(self.dataset_path).mkdir(parents=True, exist_ok=True)
            dataset = requests.get('https://forecasters.org/data/m3comp/M3C.xls')
            with open(self.dataset_file, 'wb') as f:
                f.write(dataset.content)

        self.sheet_map = {
            'yearly': 'M3Year',
            'quarterly': 'M3Quart',
            'monthly': 'M3Month',
            'others': 'M3Other',
        }
        self.frequency_map = {
            'yearly': '12M',
            'quarterly': '3M',
            'monthly': 'M',
            'others': 'H'
        }

    def load(self, frequency: str, subset_filter: str, training: bool) -> ListDataset:
        """
        Load M3 dataset.

        :param frequency: Frequency (gluonts format).
        :param subset_filter: one of [yearly, quarterly, monthly, others]
        :param training: Whole dataset if False. Without last horizon if True.
        :return: ListDataset (gluonts format).
        """
        meta_columns = 6
        dataset = pd.read_excel(self.dataset_file, sheet_name=self.sheet_map[subset_filter])

        def extract_date(row):
            year = row['Starting Year'] if subset_filter != 'others' else 1
            month = 1
            day = 1
            if subset_filter == 'quarterly':
                month = row['Starting Quarter'] * 3 - 2
            elif subset_filter == 'monthly':
                month = row['Starting Month']
            if month < 1 or month > 12:
                month = 1
            if year < 1:
                year = 1
            return pd.to_datetime(f'{year}-{month}-{day}')

        def holdout(row) -> int:
            return row['NF'] if training else 0

        items_all = [{
            'item_id': row['Series'],
            'start': extract_date(row),
            'horizon': row['NF'],
            'target': row.values[meta_columns:meta_columns + row['N'] - holdout(row)]}
            for _, row in dataset.iterrows()]

        return ListDataset(items_all, freq=frequency)

    def evaluate_subset(self, forecast: np.ndarray, subset_filter: str) -> np.ndarray:
        target_dataset = self.load(frequency=self.frequency_map[subset_filter],
                                   subset_filter=subset_filter,
                                   training=False)

        target = np.array([x['target'][-x['horizon']:] for x in target_dataset])
        return 200 * np.abs(forecast - target) / (forecast + target)

    def evaluate(self, forecast: np.ndarray) -> Dict[str, float]:
        """
        Evaluate forecast.

        :param forecast: Forecast to evaluate.
        :return: Scores for each forecast point
        """
        score_sum = 0
        time_points = 0
        scores = [
            self.evaluate_subset(forecast[:645, :6], 'yearly'),
            self.evaluate_subset(forecast[645:1401, :8], 'quarterly'),
            self.evaluate_subset(forecast[1401:2829, :18], 'monthly'),
            self.evaluate_subset(forecast[2829:3003, :8], 'others'),
        ]
        for scores_group in scores:
            for ts_scores in scores_group:
                score_sum += np.sum(ts_scores)
                time_points += len(ts_scores)
        return {
                   'yearly': float(np.mean(scores[0])),
                   'quarterly': float(np.mean(scores[1])),
                   'monthly': float(np.mean(scores[2])),
                   'others': float(np.mean(scores[3])),
                   'average': float(np.mean(np.array(score_sum / time_points)))
        }
