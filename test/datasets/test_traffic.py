"""
Traffic summary unit test
"""
import unittest

import numpy as np

from datasets.traffic import TrafficDataset


class TestTrafficEvaluation(unittest.TestCase):
    def test_evaluation(self):
        dataset = TrafficDataset()

        training_range = '2008-01-02 01:2009-03-24 01'
        training_values = TrafficDataset.to_array(dataset.load(frequency='H',
                                                               subset_filter=training_range,
                                                               training=True))
        test_values = TrafficDataset.to_array(dataset.load(frequency='H',
                                                           subset_filter=training_range,
                                                           training=False))

        mean_forecasts = []
        for i in range(7):
            window_training_set = np.concatenate(
                [training_values, test_values[:, :i * 24]], axis=1)
            window_forecast = np.repeat(
                np.repeat(np.mean(window_training_set), training_values.shape[0], axis=0)[:, None],
                24, axis=1)
            mean_forecasts = window_forecast if len(mean_forecasts) == 0 else np.concatenate(
                [mean_forecasts, window_forecast], axis=1)
        mean_forecasts = np.tile(mean_forecasts, reps=(3, 1))  # mock other splits
        score = dataset.evaluate(mean_forecasts)

        # as per Table 2 in https://www.cs.utexas.edu/~rofuyu/papers/tr-mf-nips.pdf
        self.assertTrue(np.round(score['last7days'], 2), 0.56)


if __name__ == '__main__':
    unittest.main()
