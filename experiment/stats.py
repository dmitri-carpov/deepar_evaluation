"""
Experiment statistics
"""
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import STORAGE_PATH
from experiment.config import CONFIGURATIONS


class Stats:
    def __init__(self,
                 source_dataset_name: str,
                 target_dataset_name,
                 subsets_mapping: List[Tuple[str, str]],
                 model_name: str,
                 experiments: int):
        self.source_dataset_name = source_dataset_name
        self.target_dataset_name = target_dataset_name
        self.subsets_mapping = subsets_mapping
        self.model_name = model_name
        self.experiments = experiments
        self.dataset = CONFIGURATIONS['datasets'][source_dataset_name]()
        self.target_dataset = CONFIGURATIONS['datasets'][target_dataset_name]()

    def scores(self, ensemble_size: int) -> Dict[str, float]:
        ensemble_files = np.random.choice(list(range(self.experiments)),
                                          size=ensemble_size, replace=False)
        ensemble_forecasts = []
        for source_subset, target_subset in self.subsets_mapping:
            forecasts_dir = os.path.join(STORAGE_PATH,
                                         self.model_name,
                                         f'{self.source_dataset_name}{source_subset}_{self.target_dataset_name}{target_subset}',
                                         'forecasts')
            subset_forecasts = []
            for ensemble_file in ensemble_files:
                forecast_file = os.path.join(forecasts_dir, f'{ensemble_file}.npy')
                subset_forecasts.append(np.load(forecast_file))
            ensemble_forecasts.extend(np.median(np.array(subset_forecasts), axis=0))
        return self.target_dataset.evaluate(pd.DataFrame(ensemble_forecasts).values)
