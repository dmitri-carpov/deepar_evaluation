"""
Tourism Dataset
"""
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import patoolib
import requests
from gluonts.dataset.common import ListDataset

from config import DATASETS_PATH
from datasets.dataset import Dataset


class TourismDataset(Dataset):
    """
    Tourism Dataset
    """

    def __init__(self):
        self.dataset_path = os.path.join(DATASETS_PATH, 'tourism')
        ready_flag = os.path.join(self.dataset_path, '_SUCCESS')
        if not os.path.isfile(ready_flag):
            Path(self.dataset_path).mkdir(parents=True, exist_ok=True)
            dataset = requests.get('https://robjhyndman.com/data/27-3-Athanasopoulos1.zip')
            dataset_archive = os.path.join(self.dataset_path, '27-3-Athanasopoulos1.zip')
            with open(dataset_archive, 'wb') as f:
                f.write(dataset.content)
            patoolib.extract_archive(dataset_archive, outdir=self.dataset_path)
            Path(ready_flag).touch()

        self.frequency_map = {
            'yearly': '12M',
            'quarterly': '3M',
            'monthly': 'M'
        }

        self.horizon_map = {
            'yearly': 4,
            'quarterly': 8,
            'monthly': 24
        }

    def load(self, frequency: str, subset_filter: str, training: bool) -> ListDataset:
        """
        Load tourism dataset.

        :param frequency:
        :param subset_filter:
        :param training:
        :return:
        """
        ds = pd.read_csv(os.path.join(self.dataset_path, f'{subset_filter}_in.csv'),
                         header=0, delimiter=",").T

        def extract_date(row):
            year = int(row[1])
            month = 1
            day = 1
            if subset_filter == 'quarterly':
                month = int(row[2] * 3 - 2)
            elif subset_filter == 'monthly':
                month = int(row[2])

            if month < 1 or month > 12:
                month = 1
            if year < 1:
                year = 1
            return pd.to_datetime(f'{year}-{month}-{day}')

        ds_dict = []
        meta_columns = 2 if subset_filter == 'yearly' else 3
        for item_id, row in ds.iterrows():
            ds_dict.append({
                'item_id': item_id,
                'start': extract_date(row),
                'horizon': self.horizon_map[subset_filter],
                'target': row.values[meta_columns:meta_columns + int(row[0])]
            })

        if not training:
            ds_o = pd.read_csv(os.path.join(self.dataset_path, f'{subset_filter}_oos.csv'),
                               header=0, delimiter=",").T
            i = 0
            for item_id, row in ds_o.iterrows():
                assert (ds_dict[i]['item_id'] == item_id)
                ds_dict[i]['target'] = np.concatenate([ds_dict[i]['target'],
                                                       row.values[meta_columns:meta_columns + int(row[0])]])
                i += 1

        return ListDataset(ds_dict, freq=self.frequency_map[subset_filter])

    def evaluate_subset(self, forecast: np.ndarray, subset_filter: str) -> np.ndarray:
        target_dataset = self.load(frequency=self.frequency_map[subset_filter],
                                   subset_filter=subset_filter,
                                   training=False)
        target = np.array([x['target'][-x['horizon']:] for x in target_dataset])
        return 100 * np.abs(forecast - target) / target

    def evaluate(self, forecast: np.ndarray) -> Dict[str, float]:
        """
        Evaluate tourism forecasts.

        :param forecast:
        :param subset_filter:
        :return:
        """
        score_sum = 0
        time_points = 0
        scores = [
            self.evaluate_subset(forecast[:518, :4], 'yearly'),
            self.evaluate_subset(forecast[518:945, :8], 'quarterly'),
            self.evaluate_subset(forecast[945:1311, :24], 'monthly')
        ]
        for scores_group in scores:
            for ts_scores in scores_group:
                score_sum += np.sum(ts_scores)
                time_points += len(ts_scores)
        return {
            'yearly': float(np.mean(scores[0])),
            'quarterly': float(np.mean(scores[1])),
            'monthly': float(np.mean(scores[2])),
            'average': float(np.mean(np.array(score_sum / time_points)))
        }
