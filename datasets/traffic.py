"""
Traffic Dataset
"""
import os
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import patoolib
import requests
from dateutil.relativedelta import relativedelta
from gluonts.dataset.common import ListDataset
from tqdm import tqdm

from config import DATASETS_PATH
from datasets.dataset import Dataset


class TrafficDataset(Dataset):
    """
    Traffic Dataset.
    """
    def __init__(self):
        self.dataset_path = os.path.join(DATASETS_PATH, 'traffic')
        self.cache_file = os.path.join(self.dataset_path, 'data.npy')
        self.dates_file = os.path.join(self.dataset_path, 'dates.npy')
        if not os.path.isfile(self.cache_file):
            Path(self.dataset_path).mkdir(parents=True, exist_ok=True)
            dataset_archive = os.path.join(self.dataset_path, 'PEMS-SF.zip')
            if not os.path.isfile(dataset_archive):
                dataset = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00204/PEMS-SF.zip')
                with open(dataset_archive, 'wb') as f:
                    f.write(dataset.content)
            patoolib.extract_archive(dataset_archive, outdir=self.dataset_path)
            with open(os.path.join(self.dataset_path, 'PEMS_train'), 'r') as f:
                train_raw_data = f.readlines()
            with open(os.path.join(self.dataset_path, 'PEMS_test'), 'r') as f:
                test_raw_data = f.readlines()
            with open(os.path.join(self.dataset_path, 'randperm'), 'r') as f:
                permutations = f.readlines()
            permutations = np.array(permutations[0].rstrip()[1:-1].split(' ')).astype(np.int)

            raw_data = train_raw_data + test_raw_data

            # The assumption below does NOT affect the results, because the splits we use in the publication are
            # based on either dates within the first 6 months, where the labels are aligned or on
            # the last values of dataset. Thus there should not be any confusion with misaligned split points.
            #
            # Dataset dates issue:
            #
            # From the dataset description [https://archive.ics.uci.edu/ml/datasets/PEMS-SF] :
            # "The measurements cover the period from Jan. 1st 2008 to Mar. 30th 2009"
            # and
            # "We remove public holidays from the dataset, as well
            # as two days with anomalies (March 8th 2009 and March 9th 2008)".
            #
            # Based on provided labels, which are days of week, the sequence of days had only 10 gaps by 1 day,
            # where the first 6 correspond to a holiday or anomalous day, but the other 4 gaps happen on "random" dates,
            # meaning we could not find any holiday or the mentioned anomalous days around those dates.
            #
            # More over, the number of days between 2008-01-01 and 2009-03-30 is 455, with only 10 gaps it's
            # not possible to fill dates up to 2009-03-30, it should be 15 gaps (if 2009-01-01 is included, 14 otherwise).
            #
            # Thus, it is not clear if either labels are not correct or the dataset description.
            #
            # Since we are not using any covariates and the split dates after the first 6 months we just fill the gaps with
            # the most common holidays, it does not have any impact on the split points anyway.
            current_date = datetime.strptime('2008-01-01', '%Y-%m-%d')
            excluded_dates = [
                datetime.strptime('2008-01-01', '%Y-%m-%d'),
                datetime.strptime('2008-01-21', '%Y-%m-%d'),
                datetime.strptime('2008-02-18', '%Y-%m-%d'),
                datetime.strptime('2008-03-09', '%Y-%m-%d'),
                datetime.strptime('2008-05-26', '%Y-%m-%d'),
                datetime.strptime('2008-07-04', '%Y-%m-%d'),
                datetime.strptime('2008-09-01', '%Y-%m-%d'),
                datetime.strptime('2008-10-13', '%Y-%m-%d'),
                datetime.strptime('2008-11-11', '%Y-%m-%d'),
                datetime.strptime('2008-11-27', '%Y-%m-%d'),
                datetime.strptime('2008-12-25', '%Y-%m-%d'),
                datetime.strptime('2009-01-01', '%Y-%m-%d'),
                datetime.strptime('2009-01-19', '%Y-%m-%d'),
                datetime.strptime('2009-02-16', '%Y-%m-%d'),
                datetime.strptime('2009-03-08', '%Y-%m-%d'),
            ]
            dates = []
            np_array = []
            for i in tqdm(range(len(permutations))):
                # values
                matrix = raw_data[np.where(permutations == i + 1)[0][0]].rstrip()[1:-1]
                daily = []
                for row_vector in matrix.split(';'):
                    daily.append(np.array(row_vector.split(' ')).astype(np.float32))
                daily = np.array(daily)
                if len(np_array) == 0:
                    np_array = daily
                else:
                    np_array = np.concatenate([np_array, daily], axis=1)

                # dates
                while current_date in excluded_dates:  # skip those in excluded dates
                    current_date = current_date + timedelta(days=1)
                dates.extend([pd.to_datetime((current_date + timedelta(hours=i + 1)).strftime('%Y-%m-%d %H'))
                              for i in range(24)])
                current_date = current_date + timedelta(days=1)

            # aggregate 10 minutes events to hourly
            hourly = np.array([list(map(np.mean, zip(*(iter(lane),) * 6))) for lane in tqdm(np_array)])
            hourly.dump(self.cache_file)
            np.array(dates).dump(self.dates_file)
        self.values = np.load(self.cache_file, allow_pickle=True)
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

        :param forecast: forecasts for "2008-01-14", "2008-06-15", "2009-03-24 01" concatenated along axis 0
        :return:
        """
        return {
            '2008-01-14': float(self.evaluate_subset(forecast[:963], '2008-01-14 00')),
            '2008-06-15': float(self.evaluate_subset(forecast[963:1926], '2008-06-15 00')),
            'last7days': float(self.evaluate_subset(forecast[1926:2889], '2009-03-24 01'))
        }

    def _dates_to_index(self, from_date, to_date) -> range:
        from_index = int(np.argwhere(self.dates == from_date)[0])
        to_index = int(np.argwhere(self.dates == to_date)[0]) if to_date <= max(self.dates) else len(self.dates)
        return range(from_index, to_index)