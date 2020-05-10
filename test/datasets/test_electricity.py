"""
Electricity summary unit test
"""
import unittest

import numpy as np

from datasets.electricity import ElectricityDataset


class TestElectricityEvaluation(unittest.TestCase):
    def test_evaluation(self):
        dataset = ElectricityDataset()

        training_range = '2012-01-01 01:2014-12-25 01'
        training_values = ElectricityDataset.to_array(dataset.load(frequency='H',
                                                                   subset_filter=training_range,
                                                                   training=True))
        test_values = ElectricityDataset.to_array(dataset.load(frequency='H',
                                                               subset_filter=training_range,
                                                               training=False))

        mean_forecasts = []
        for i in range(7):
            window_training_set = np.concatenate(
                [training_values, test_values[:, :i * 24]], axis=1)
            window_forecast = np.repeat(
                np.repeat(np.mean(window_training_set), 370, axis=0)[:, None],
                24, axis=1)
            mean_forecasts = window_forecast if len(mean_forecasts) == 0 else np.concatenate(
                [mean_forecasts, window_forecast], axis=1)
        mean_forecasts = np.tile(mean_forecasts, reps=(3, 1)) # mock other splits
        score = dataset.evaluate(mean_forecasts)

        # as per Table 2 in https://www.cs.utexas.edu/~rofuyu/papers/tr-mf-nips.pdf
        self.assertEqual(np.round(score['last7days'], 3), 1.410)


if __name__ == '__main__':
    unittest.main()
