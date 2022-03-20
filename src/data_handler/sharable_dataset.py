import sharedmem
from gluonts.dataset.common import ListDataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper


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
             }], freq=self.__freq)
        return dataset

    @staticmethod
    def _share(array):
        if isinstance(array, sharedmem.sharedmem.anonymousmemmap):
            return array
        else:
            fp = sharedmem.empty(array.shape, dtype=array.dtype)
            fp[:] = array[:]
            return fp


class SharableMultiVariateDataset:
    """
    SharableMultiVariateDataset  allow storing of multiple sharable
        target arrays and allow convertion to grouped_list_dataset.

        - [X] Convert the multivariate_dataset into local grouped dataset using train_grouper and test_grouper

    """

    def __init__(self, sharable_list_datasets, freq='D'):
        for i in range(len(sharable_list_datasets)):
            assert sharable_list_datasets[i].freq == freq
        self.__sharable_list_datasets = sharable_list_datasets

    def to_local(self, mode=None):
        assert mode == 'train' or mode == 'test' or mode is None
        ts_jsons = []
        for sharable_list_dataset in self.__sharable_list_datasets:
            ts_jsons.append(
                {
                    "start": sharable_list_dataset.start,
                    "target": sharable_list_dataset.target,
                }
            )
        dataset = ListDataset(ts_jsons, freq='D')
        grouper = MultivariateGrouper(
            max_target_dim=len(
                self.__sharable_list_datasets))
        grouped_dataset = grouper(dataset)
        return grouped_dataset
