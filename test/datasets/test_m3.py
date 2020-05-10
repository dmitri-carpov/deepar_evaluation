"""
M3 evaluation unit test
"""
import os
import tempfile
import unittest
from typing import Optional

import numpy as np
import pandas as pd
import requests

from datasets.m3 import M3Dataset

class TestM3Evaluation(unittest.TestCase):
    DATASET_DIR = tempfile.mkdtemp()
    FORECASTS_PATH = os.path.join(DATASET_DIR, 'M3Forecast.xls')
    def setUp(self) -> None:
        if not os.path.isfile(self.FORECASTS_PATH):
            forecasts = requests.get('https://forecasters.org/data/m3comp/M3Forecast.xls')
            with open(self.FORECASTS_PATH, 'wb') as f:
                f.write(forecasts.content)

    def test_evaluation(self):
        m3_dataset = M3Dataset()
        naive2 = pd.read_excel(self.FORECASTS_PATH, sheet_name='NAIVE2', header=None).values
        scores = m3_dataset.evaluate(naive2[:, 2:])

        self.assertEqual(np.round(scores['yearly'], 2), 17.88)
        self.assertEqual(np.round(scores['quarterly'], 2), 9.95)
        self.assertEqual(np.round(scores['monthly'], 2), 16.91)
        self.assertEqual(np.round(scores['others'], 2), 6.3)
        self.assertEqual(np.round(scores['average'], 2), 15.47)

if __name__ == '__main__':
    unittest.main()
