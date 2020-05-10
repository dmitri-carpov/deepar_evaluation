"""
Pipeline basic functions.
"""
import os
from pathlib import Path

import numpy as np
from gluonts.dataset.common import ListDataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.predictor import Predictor
from gluonts.trainer import Trainer


def train(dataset: ListDataset,
          frequency: str,
          horizon: int,
          model_name: str,
          num_layers: int,
          num_cells: int,
          epochs: int,
          patience: int,
          weight_decay: float,
          dropout_rate: float,
          batch_size: int,
          snapshot_dir: str,
          overwrite: bool):
    """
    Train a model.

    :param dataset:
    :param model_name:
    :param horizon:
    :param frequency:
    :param snapshot_dir:
    :param epochs:
    :param patience:
    :param weight_decay:
    :param batch_size:
    :param dropout_rate:
    :param num_layers:
    :param num_cells:
    :param overwrite:
    :return:
    """
    model_dir = Path(snapshot_dir)
    if not overwrite and os.path.isdir(snapshot_dir):
        return Predictor.deserialize(model_dir)
    trainer = Trainer(epochs=epochs, patience=patience, weight_decay=weight_decay, batch_size=batch_size)
    if model_name == 'deepar':
        estimator = DeepAREstimator(freq=frequency,
                                    scaling=False,
                                    dropout_rate=dropout_rate,
                                    num_layers=num_layers,
                                    num_cells=num_cells,
                                    prediction_length=horizon,
                                    trainer=trainer)
    else:
        raise Exception(f'Unknown model {model_name}')
    predictor = estimator.train(training_data=dataset)
    model_dir.mkdir(parents=True, exist_ok=overwrite)
    predictor.serialize(model_dir)
    return predictor


def predict(dataset: ListDataset,
            snapshot_dir: str,
            samples: int=100):
    """
    Make predictions using model snapshot.
    :param dataset:
    :param snapshot_dir:
    :param samples:
    :return:
    """

    predictor = Predictor.deserialize(Path(snapshot_dir))
    forecast_it, ts_it = make_evaluation_predictions(dataset,
                                                     predictor=predictor,
                                                     num_samples=samples)
    return np.array([np.median(x.samples, axis=0) for x in list(iter(forecast_it))])

def scale(dataset: ListDataset) -> ListDataset:
    entries = []
    freq = None
    for entry in dataset:
        if freq is None:
            freq = entry['start'].freq
        values = entry['target']
        scaling_factor = np.abs(np.max(values))
        entries.append({**entry, 'target': values / scaling_factor, 'scaling_factor': scaling_factor})
    return ListDataset(entries, freq=freq)

def unscale(dataset: ListDataset, forecast: np.ndarray) -> np.ndarray:
    scaling_factors = np.array([r['scaling_factor'] for r in dataset])
    return forecast * scaling_factors[:, None]
