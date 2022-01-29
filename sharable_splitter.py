import pandas as pd
import numpy as np
from datetime import timedelta
from sharable_dataset import SharableListDataset


class Splitter:
        def __init__(self, split_date):
                self.__split_date = split_date
        def split(self, shm_dataset):
                assert shm_dataset.freq == 'D'
                n_begin_days = (self.__split_date - shm_dataset.start).days
                dataset1 = SharableListDataset(
                        shm_dataset.start, 
                        shm_dataset.target[:n_begin_days], 
                        freq=shm_dataset.freq)
                dataset2 = SharableListDataset(
                        self.__split_date,
                        shm_dataset.target[n_begin_days:],
                        freq=shm_dataset.freq
                )
                return dataset1, dataset2

if __name__ == '__main__':
        shm_dataset = SharableListDataset(
                pd.Timestamp('2021-01-01', freq='D'),
                np.arange(20),
                freq='D'
        )
        print(shm_dataset)
        print(list(shm_dataset.to_local())[0])
        splitter = Splitter(pd.Timestamp('2021-01-03', freq='D'))
        train, test = splitter.split(shm_dataset)
        print(train.target)
        print(test.target)

        print(list(train.to_local())[0])
        print(list(test.to_local())[0])
