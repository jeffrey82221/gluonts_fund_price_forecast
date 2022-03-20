import pandas as pd
import numpy as np
from src.data_handler.sharable_dataset import SharableListDataset


class Splitter:
    def __init__(self, split_date, with_overlap=True):
        self.__split_date = split_date
        self.__with_overlap = with_overlap

    def split(self, shm_dataset):
        assert shm_dataset.freq == 'D'
        n_begin_days = (self.__split_date - shm_dataset.start).days
        train = SharableListDataset(
            shm_dataset.start,
            shm_dataset.target[:n_begin_days],
            freq=shm_dataset.freq)
        test = SharableListDataset(
            shm_dataset.start if self.__with_overlap else self.__split_date,
            shm_dataset.target if self.__with_overlap else shm_dataset.target[n_begin_days:],
            freq=shm_dataset.freq
        )
        return train, test


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
