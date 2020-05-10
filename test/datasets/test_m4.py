"""
M4 evaluation unit test
"""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import requests
import patoolib

from datasets.m4 import M4Dataset

class TestM4Evaluation(unittest.TestCase):
    DATASET_DIR = tempfile.mkdtemp()
    FORECASTS_PATH = os.path.join(DATASET_DIR, 'submission-118.csv')
    def setUp(self) -> None:
        if not os.path.isfile(self.FORECASTS_PATH):
            forecasts = requests.get('https://github.com/M4Competition/M4-methods/raw/master/Point%20Forecasts/submission-118.rar')
            archive_file = os.path.join(self.DATASET_DIR, 'submission-118.rar')
            if not os.path.isfile(archive_file):
                forecasts = requests.get(
                    'https://github.com/M4Competition/M4-methods/raw/master/Point%20Forecasts/submission-118.rar')
                with open(archive_file, 'wb') as f:
                    f.write(forecasts.content)
                patoolib.extract_archive(archive_file, outdir=self.DATASET_DIR)

    def test_evaluation(self):
        dataset = M4Dataset()
        m4_winner_forecast = pd.read_csv(self.FORECASTS_PATH)
        m4_winner_forecast.set_index(m4_winner_forecast.columns[0], inplace=True)
        scores = dataset.evaluate(m4_winner_forecast.values)

        # Results are based on Table 1 of:
        # https://www.researchgate.net/profile/Spyros_Makridakis/publication/325901666_The_M4_Competition_Results_
        # findings_conclusion_and_way_forward/links/5b2c9aa4aca2720785d66b5e/The-M4-Competition-Results-findings-
        # conclusion-and-way-forward.pdf?origin=publication_detail
        self.assertEqual(np.round(scores['yearly'], 3), 13.176)
        self.assertEqual(np.round(scores['quarterly'], 3), 9.679)
        self.assertEqual(np.round(scores['monthly'], 3), 12.126)
        self.assertEqual(np.round(scores['others'], 3), 4.013)
        self.assertEqual(np.round(scores['average'], 3), 11.374)

if __name__ == '__main__':
    unittest.main()
