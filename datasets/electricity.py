"""
Electricity Dataset
"""
import os
import requests
from datetime import datetime
from pathlib import Path
from dateutil.relativedelta import relativedelta
from typing import Dict

import numpy as np
import pandas as pd
import patoolib
from gluonts.dataset.common import ListDataset
from tqdm import tqdm

from datasets.dataset import Dataset
from config import DATASETS_PATH

class ElectricityDataset(Dataset):
    """
    Electricity Dataset.
    """
    def __init__(self):
        self.dataset_path = os.path.join(DATASETS_PATH, 'electricity')
        self.cache_file = os.path.join(self.dataset_path, 'data.npy')
        self.dates_file = os.path.join(self.dataset_path, 'dates.npy')
        if not os.path.isfile(self.cache_file):
            Path(self.dataset_path).mkdir(parents=True, exist_ok=True)
            dataset_archive = os.path.join(self.dataset_path, 'LD2011_2014.txt.zip')
            if not os.path.isfile(dataset_archive):
                dataset = requests.get(
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip')
                with open(dataset_archive, 'wb') as f:
                    f.write(dataset.content)
            patoolib.extract_archive(dataset_archive, outdir=self.dataset_path)
            parsed_file = os.path.join(self.dataset_path, 'parsed.txt')
            with open(os.path.join(self.dataset_path, 'LD2011_2014.txt'), 'r') as f:
                raw = f.read()
            with open(parsed_file, 'w') as f:
                f.write(raw.replace(',', '.'))
            raw_df = pd.read_csv(parsed_file, delimiter=';')
            data = raw_df[raw_df.columns[1:]].values[365 * 24 * 4:, :]
            dates = raw_df[raw_df.columns[0]].values[365 * 24 * 4:]

            # aggregate to hourly
            aggregated = []
            for i in tqdm(range(0, data.shape[0], 4)):
                aggregated.append(data[i:i + 4, :].sum(axis=0))
            aggregated = np.array(aggregated)
            dataset = aggregated.T  # use time step as second dimension.
            dates = np.unique(list(
                map(lambda s: pd.to_datetime(datetime.strptime(s, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H')),
                    dates)))[1:]

            np.save(self.dates_file, dates)
            np.save(self.cache_file, dataset)
        self.values = np.load(self.cache_file)
        self.dates = np.load(self.dates_file, allow_pickle=True)

    def load(self, frequency: str, subset_filter: str, training: bool) -> ListDataset:
        """
        Load electricity dataset.

        :param frequency:
        :param subset_filter: dates as "from_date:to_date" in "YYYY-mm-dd H" format.
        :param training: If False then to_date will be extended to 7 days in future.
        :return:
        """
        dates = subset_filter.split(':')
        from_date = pd.to_datetime(dates[0])
        to_date = pd.to_datetime(dates[1])
        if not training:
            to_date = to_date + relativedelta(hours=24 * 7)

        items_all = [{
            'item_id': i,
            'start': from_date,
            'horizon': 24,
            'target': values}
            for i, values in enumerate(self.values[:, self._dates_to_index(from_date, to_date)])]

        return ListDataset(items_all, freq=frequency)

    def evaluate_subset(self, forecast: np.ndarray, subset_filter: str) -> np.ndarray:
        # ignore first date because it indicates start of the training set
        from_date = pd.to_datetime(subset_filter)
        to_date = pd.to_datetime(from_date) + relativedelta(hours=forecast.shape[1])
        target = self.values[:, self._dates_to_index(from_date, to_date)]
        return np.mean(np.abs(target - forecast)) / np.mean(np.abs(target))

    def evaluate(self, forecast: np.ndarray) -> Dict[str, float]:
        """
        Evaluate electricity forecasts.

        :param forecast: forecasts for "2014-03-31 00", "2014-09-01 00", "2014-12-25 01" concatenated along axis 0
        :return:
        """
        return {
            '2014-03-31': float(self.evaluate_subset(forecast[:370], '2014-03-31 00')),
            '2014-09-01': float(self.evaluate_subset(forecast[370:740], '2014-09-01 00')),
            'last7days': float(self.evaluate_subset(forecast[740:1110], '2014-12-25 01'))
        }

    def _dates_to_index(self, from_date, to_date) -> range:
        from_index = int(np.argwhere(self.dates == from_date)[0])
        to_index = int(np.argwhere(self.dates == to_date)[0]) if to_date <= max(self.dates) else len(self.dates)
        return range(from_index, to_index)