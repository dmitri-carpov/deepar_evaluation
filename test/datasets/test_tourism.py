"""
Tourism summary unit test
"""
import unittest

import numpy as np
import pandas as pd

from datasets.tourism import TourismDataset


class TestTourismEvaluation(unittest.TestCase):
    def test_evaluation(self):
        dataset = TourismDataset()

        seasonality = {'yearly': 1, 'quarterly': 4, 'monthly': 12}
        forecasts = []
        for subset_filter in ['yearly', 'quarterly', 'monthly']:
            for entry in dataset.load(frequency=dataset.frequency_map[subset_filter],
                                      subset_filter=subset_filter,
                                      training=True):
                horizon = entry['horizon']
                insample = entry['target']
                forecast = np.zeros((horizon,))
                for i in range(horizon):
                    idx = len(insample) + i % seasonality[subset_filter] - seasonality[subset_filter]
                    forecast[i] = insample[idx]
                forecasts.append(forecast)
        scores = dataset.evaluate(pd.DataFrame(forecasts).values)

        # From https://robjhyndman.com/papers/forecompijf.pdf
        self.assertTrue(np.round(scores['yearly'], 2) == 23.61)
        self.assertTrue(np.round(scores['quarterly'], 2) == 16.46)
        self.assertTrue(np.round(scores['monthly'], 2) == 22.56)

if __name__ == '__main__':
    unittest.main()
