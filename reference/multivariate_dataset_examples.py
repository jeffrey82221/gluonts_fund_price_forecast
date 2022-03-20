'''
Organize nav curves of multiple funds into the multi-timeseries objects offered by gluonts.
'''
import os
import inspect
import sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from gluonts.dataset.common import ListDataset
import numpy as np
import pandas as pd
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import get_dataset

# exchange_rate_nips, electricity_nips, traffic_nips, solar_nips,
# wiki-rolling_nips, ## taxi_30min is buggy still
dataset = get_dataset("electricity_nips", regenerate=True)

train_grouper = MultivariateGrouper(max_target_dim=min(
    2000, int(dataset.metadata.feat_static_cat[0].cardinality)))

test_grouper = MultivariateGrouper(num_test_dates=int(len(dataset.test) / len(dataset.train)),
                                   max_target_dim=min(2000, int(dataset.metadata.feat_static_cat[0].cardinality)))

dataset_train = train_grouper(dataset.train)
dataset_test = test_grouper(dataset.test)
print('Example Success')

ts_jsons = []
for i in range(10):
    ts_jsons.append(
        {
            "start": pd.Timestamp('2021-01-01', freq='D'),
            "target": np.arange(300 + i),
        }
    )
dataset = ListDataset(ts_jsons, freq='D')
print(next(iter(dataset)))
train_grouper = MultivariateGrouper(max_target_dim=10)
grouped_dataset = train_grouper(dataset)
print(len(grouped_dataset))
print(next(iter(grouped_dataset)))
print('Own version success')
