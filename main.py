"""
Experiment logic.
"""
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from fire import Fire

from config import STORAGE_PATH
from experiment.config import CONFIGURATIONS
from experiment.pipeline import train, predict, scale, unscale


def datasets():
    from datasets.m3 import M3Dataset
    from datasets.tourism import TourismDataset
    from datasets.electricity import ElectricityDataset
    from datasets.traffic import TrafficDataset

    # Download datasets if they do not exist
    M3Dataset()
    TourismDataset()
    ElectricityDataset()
    TrafficDataset()

def run(source_dataset: str,
        source_subset: str,
        target_dataset: str,
        target_subset: str,
        frequency: str,
        horizon: int,
        model_name: str,
        experiments: int):
    experiment_name = f'{source_dataset}{source_subset}_{target_dataset}{target_subset}'
    experiment_path = os.path.join(STORAGE_PATH, model_name, experiment_name)
    Path(experiment_path).mkdir(parents=True, exist_ok=True)

    snapshot_dir = os.path.join(experiment_path, 'model')
    Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
    forecasts_dir = os.path.join(experiment_path, 'forecasts')
    Path(forecasts_dir).mkdir(parents=True, exist_ok=True)

    dataset = CONFIGURATIONS['datasets'][source_dataset]()
    target_dataset = CONFIGURATIONS['datasets'][target_dataset]()
    model_params = CONFIGURATIONS['models'][model_name][source_subset]
    print(model_params)

    finished_experiments = len(os.listdir(forecasts_dir))
    for experiment_number in range(finished_experiments, experiments):
        scaled_dataset = scale(dataset.load(frequency=frequency,
                                            subset_filter=source_subset,
                                            training=True))
        train(dataset=scaled_dataset,
              frequency=frequency,
              horizon=horizon,
              model_name=model_name,
              num_layers=model_params['num_layers'],
              num_cells=model_params['num_cells'],
              epochs=model_params['epochs'],
              patience=model_params['patience'],
              weight_decay=model_params['weight_decay'],
              dropout_rate=model_params['dropout_rate'],
              batch_size=model_params['batch_size'],
              snapshot_dir=snapshot_dir,
              overwrite=True)

        if target_dataset == 'Electricity' or target_dataset == 'Traffic':
            # rolling forecast
            start_date = target_subset.split(':')[0]
            test_start = pd.to_datetime(target_subset.split(':')[1])
            forecasts = []
            for i in range(7):
                end_date = (test_start + relativedelta(hours=i * 24)).strftime('%Y-%m-%d %H')
                input_set = target_dataset.load(frequency=frequency,
                                                subset_filter=f'{start_date}:{end_date}',
                                                training=True)
                input_set = scale(input_set)
                forecast = predict(dataset=input_set,
                                   snapshot_dir=snapshot_dir,
                                   samples=100)
                forecast = unscale(input_set, forecast)
                forecasts.append(forecast)
            forecasts = np.concatenate(forecasts, axis=1)
        else:
            input_set = target_dataset.load(frequency=frequency, subset_filter=target_subset, training=False)
            input_set = scale(input_set)
            forecasts = predict(dataset=input_set, snapshot_dir=snapshot_dir, samples=100)
            forecasts = unscale(input_set, forecasts)

        np.save(os.path.join(forecasts_dir, f'{str(experiment_number)}.npy'), forecasts)


if __name__ == '__main__':
    logging.root.setLevel(logging.INFO)
    Fire()

