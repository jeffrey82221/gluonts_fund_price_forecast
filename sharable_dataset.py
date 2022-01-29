from gluonts.dataset.common import ListDataset
import sharedmem

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