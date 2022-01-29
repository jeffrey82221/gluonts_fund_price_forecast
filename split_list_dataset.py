import pandas as pd
import numpy as np
from gluonts.dataset.common import ListDataset
from gluonts.dataset.split.splitter import DateSplitter
import sharedmem
from datetime import timedelta


class SharableListDataset:
        def __init__(self, start, target, freq='D'):
                self.__start = start

                self.__target = SharableListDataset._share(target)
                self.__freq = freq
        @property
        def start(self):
                return self.__start

        @property
        def target(self):
                return self.__target

        @property
        def freq(self):
                return self.__freq

        def to_local(self):
                dataset = ListDataset([
                        {"start": self.__start,
                        "target": self.__target,
                        }],freq=self.__freq)
                return dataset

        @staticmethod
        def _share(array):
                if isinstance(array, sharedmem.sharedmem.anonymousmemmap):
                        return array
                else:
                        fp = sharedmem.empty(array.shape, dtype=array.dtype)
                        fp[:] = array[:]
                        return fp

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
