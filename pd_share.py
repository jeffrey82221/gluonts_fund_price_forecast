"""
class storing pandas DataFrame in shared memory to be accesses by multiple processes
"""
import pandas as pd
import sharedmem
from pandas.api.types import is_datetime64_any_dtype as is_datetime


class CustomizedSharedArray:
    def __init__(self, array, callback=lambda x:x):
        fp = sharedmem.empty(array.shape, dtype=array.dtype)
        fp[:] = array[:]
        self.__fp = fp
        self.__callback = callback
        self.dtype = array.dtype
        self.shape = array.shape
    def tolist(self):
        return self.__callback(self.__fp.tolist())

    def __getitem__(self, slice):
        return self.__fp[slice]
    def __del__(self):
        del self.dtype, self.shape, self.__fp, self.__callback

class iLocIndexer:
    def __init__(self, shared_df):
        self.__shared_df = shared_df

    def __getitem__(self, slice):
        """
        Args: 
            - slice: slice object
        Returns: 
            - a new SharedDataFrame
        """
        new_shared_df = SharedDataFrame()
        new_shared_df.set_content(
            self.__shared_df.columns, 
            self.__shared_df.index[slice],
            dict([(col, self.__shared_df[col][slice]) for col in self.__shared_df.columns.tolist()])
            )
        return new_shared_df
    def __del__(self):
        del self.__shared_df

class SharedDataFrame:
    def __init__(self, dataframe=None):
        self.__values = dict()
        if dataframe is not None:
            self.set_content(
                dataframe.columns, 
                dataframe.index, 
                dict([(col, dataframe[col]) for col in dataframe.columns])
            )

    def set_content(self, columns, index, values):
        self.__set_columns(columns)
        self.__set_index(index)
        for col in values:
            self.__set_values(col, values[col])

    def __set_columns(self, columns):
        if isinstance(columns, sharedmem.sharedmem.anonymousmemmap):
            self.__cols = columns
        else:
            self.__cols = SharedDataFrame.__to_shared(columns)

    
    def __set_index(self, index):
        if isinstance(index, sharedmem.sharedmem.anonymousmemmap):
            self.__index = index
        else:
            self.__index = SharedDataFrame.__to_shared(index)

    def __set_values(self, col, col_values):
        if isinstance(col_values, sharedmem.sharedmem.anonymousmemmap):
            self.__values[col] = col_values
        else:
            self.__values[col] = SharedDataFrame.__to_shared(col_values)
            
    @property
    def iloc(self):
        return iLocIndexer(self)
    
    @staticmethod
    def __to_shared(array):
        if is_datetime(array.dtype):
            return CustomizedSharedArray(array, callback = pd.to_datetime)
        else:
            fp = sharedmem.empty(array.shape, dtype=array.dtype)
            fp[:] = array[:]
            return fp

    def __getitem__(self, col):
        return self.__values[col]

    


    @property
    def index(self):
        return self.__index

    @property
    def columns(self):
        return self.__cols

    def to_pandas(self):
        table = pd.DataFrame()
        for col in self.__cols.tolist():
            table[col] = self.__values[col].tolist()
        return table

    def __del__(self):
        for col in self.__values:
            del self.__values[col]
        del self.__cols
        del self.__index

if __name__ == '__main__':
    data = [['Y', 'N', 'N', 'N', 'N', 'N', 'N', 1],
             ['N', 'Y', 'N', 'N', 'N', 'N', 'N', 1],
             ['N', 'N', 'Y', 'N', 'N', 'N', 'N', 1],
             ['N', 'N', 'N', 'Y', 'N', 'N', 'N', 1],
             ['N', 'N', 'N', 'N', 'Y', 'N', 'N', 1],
             ['N', 'N', 'N', 'N', 'N', 'Y', 'N', 2],
             ['N', 'N', 'N', 'N', 'N', 'N', 'Y', 2],
             ['N', 'N', 'N', 'N', 'N', 'N', 'N', 3],
             ['Y', 'N', 'N', 'N', 'N', 'Y', 'Y', 1],
             ['N', 'N', 'N', 'N', 'Y', 'N', 'Y', 1]]
    df = pd.DataFrame(data=data, columns=['travel_card',
                                            'five_profession_card',
                                            'world_card',
                                            'wm_cust',
                                            'gov_employee',
                                            'military_police_firefighters',
                                            'salary_ind',
                                            'output'])
    print('Before Sharing')
    print(df)
    shared_df = SharedDataFrame(df)
    print('After Sharing')
    print(shared_df.to_pandas())
    from pandas.testing import assert_frame_equal
    assert_frame_equal(df, shared_df.to_pandas(), check_dtype=True)
    print('Identical!')
    print('Property Test:')
    print('index:', shared_df.index.tolist())
    print('columns:', shared_df.columns.tolist())
    print('iloc')
    print(shared_df['output'][2:6])
    print(shared_df.iloc[2:6]['output'])
    assert all(shared_df['output'][2:6] == shared_df.iloc[2:6]['output'])
    assert all(shared_df['world_card'][2:6] == shared_df.iloc[2:6]['world_card'])
    assert all(shared_df['five_profession_card'][2:6] == shared_df.iloc[2:6]['five_profession_card'])
    