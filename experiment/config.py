from datasets.electricity import ElectricityDataset
from datasets.m3 import M3Dataset
from datasets.m4 import M4Dataset
from datasets.tourism import TourismDataset
from datasets.traffic import TrafficDataset

CONFIGURATIONS = {
    'models': {
        'deepar': {
            'yearly': {
                'num_layers': 3,
                'num_cells': 40,
                'epochs': 300,
                'patience': 300,
                'weight_decay': 0.0,
                'dropout_rate': 0.0,
                'batch_size': 32
            },
            'quarterly': {
                'num_layers': 2,
                'num_cells': 20,
                'epochs': 100,
                'patience': 500,
                'weight_decay': 0.0,
                'dropout_rate': 0.0,
                'batch_size': 32
            },
            'monthly': {
                'num_layers': 2,
                'num_cells': 40,
                'epochs': 500,
                'patience': 500,
                'weight_decay': 0.0,
                'dropout_rate': 0.0,
                'batch_size': 32
            },
            'others': {
                'num_layers': 2,
                'num_cells': 40,
                'epochs': 100,
                'patience': 100,
                'weight_decay': 0.0,
                'dropout_rate': 0.0,
                'batch_size': 32
            },
            'weekly': {
                'num_layers': 3,
                'num_cells': 20,
                'epochs': 100,
                'patience': 100,
                'weight_decay': 0.0,
                'dropout_rate': 0.0,
                'batch_size': 32
            },
            'daily': {
                'num_layers': 3,
                'num_cells': 20,
                'epochs': 100,
                'patience': 100,
                'weight_decay': 0.0,
                'dropout_rate': 0.0,
                'batch_size': 32
            },
            'hourly': {
                'num_layers': 2,
                'num_cells': 20,
                'epochs': 50,
                'patience': 100,
                'weight_decay': 0.0,
                'dropout_rate': 0.0,
                'batch_size': 32
            },
            '2014-01-01 00:2014-03-31 00': {
                'num_layers': 2,
                'num_cells': 40,
                'epochs': 50,
                'patience': 50,
                'weight_decay': 0.0,
                'dropout_rate': 0.1,
                'batch_size': 64
            },
            '2014-01-01 00:2014-09-01 00': {
                'num_layers': 2,
                'num_cells': 40,
                'epochs': 50,
                'patience': 50,
                'weight_decay': 0.0,
                'dropout_rate': 0.1,
                'batch_size': 64
            },
            '2014-03-25 00:2014-12-25 01': {
                'num_layers': 2,
                'num_cells': 40,
                'epochs': 50,
                'patience': 50,
                'weight_decay': 0.0,
                'dropout_rate': 0.1,
                'batch_size': 64
            },
            '2008-01-02 01:2008-01-14 00': {
                'num_layers': 1,
                'num_cells': 20,
                'epochs': 5,
                'patience': 50,
                'weight_decay': 0.0,
                'dropout_rate': 0.0,
                'batch_size': 64
            },
            '2008-01-02 01:2008-06-15 00': {
                'num_layers': 4,
                'num_cells': 40,
                'epochs': 50,
                'patience': 50,
                'weight_decay': 0.0,
                'dropout_rate': 0.0,
                'batch_size': 64
            },
            '2008-09-02 01:2009-03-24 01': {
                'num_layers': 4,
                'num_cells': 40,
                'epochs': 50,
                'patience': 50,
                'weight_decay': 0.0,
                'dropout_rate': 0.0,
                'batch_size': 64
            }
        },
    },

    'datasets': {
        'M4': M4Dataset,
        'M3': M3Dataset,
        'Tourism': TourismDataset,
        'Electricity': ElectricityDataset,
        'Traffic': TrafficDataset
    }
}
