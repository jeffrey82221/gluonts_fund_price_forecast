from datetime import datetime
import pandas as pd
from gluonts.dataset.common import ListDataset
from joblib import Memory
from datetime import timedelta
from sharable_splitter import Splitter

memory = Memory('./cachedir', verbose=0)

NAV_DIR = '../fund_price_crawler/nav'

def load_dataset(file_path):
    """
    Load a nav csv and convert it to gluonts ListDataset.

    Args:
        - file_path: the path of the nav csv file
    Returns:
        - dataset: the ListDataset
    """
    nav_table = __load_raw_nav_table(file_path)
    nav_table = __fill_nav_dataframe(nav_table)
    dataset = __convert_to_list_dataset(nav_table)
    return dataset

@memory.cache
def load_nav_table(file_path):
    """
    Load the NAV csv into pandas DataFrame and fill the missing 
    values.

    Args:
        - file_path: the path of the nav csv file
    Returns: 
        - nav_table: a pandas DataFrame of nav values without missing values 
            of holiday. 
    """
    nav_table = __load_raw_nav_table(file_path)
    nav_table = __fill_nav_dataframe(nav_table)
    return nav_table

def __load_raw_nav_table(file_path):
    """
    Load the NAV csv into pandas DataFrame 

    Args: 
        - file_path: the path of the nav csv file
    Returns: 
        - nav_table: the loaded nav pandas DataFrame.
    """
    nav_table = pd.read_csv(file_path,
                            delimiter=',', header=None, index_col=False)
    nav_table.columns = ['date', 'value']
    nav_table.drop_duplicates(inplace=True)
    return nav_table

def __fill_nav_dataframe(raw_nav_table):
    """
    
    Fill in the nav values of holidays via linear interpolation. 

    Args: 
        - raw_nav_table: the raw nav dataframe loaded from the downloaded csv file
    Returns:
        - everyday_nav_table: a pandas DataFrame of nav values without missing values 
            of holiday. 
    """
    raw_nav_table.date = raw_nav_table.date.map(lambda x: datetime.strptime(x, '%Y/%m/%d'))
    idx = pd.date_range(start=raw_nav_table.date.min(),end=raw_nav_table.date.max(), freq="D")
    everyday_table = pd.DataFrame(idx, columns=['date'])
    nav_table = everyday_table.merge(raw_nav_table, how='left', on='date')
    nav_table.set_index('date', inplace=True)
    nav_table.interpolate(inplace=True)
    nav_table = nav_table.reset_index().drop_duplicates(subset='date').set_index('date')
    assert len(nav_table) == len(set(nav_table.index.tolist()))
    return nav_table

def __convert_to_list_dataset(nav_table):
    """
    Convert single nav_table (DataFrame) to GluonTS ListDataset.

    Args:
        - nav_table: the pandas table. 
    Returns:
        - dataset: the ListDataset.
    """
    dataset = ListDataset([
        {"start": nav_table.index[0],
        "target": nav_table.value}],freq="D")
    return dataset

def load_split_dataset(file_path, split_date):
    """
    Load CSV and split it into a training ListDataset
        and a testing ListDataset, according to split_date. 

    Args: 
        - file_path:  the path of the nav csv file
        - split_date: (datetime.datetime) 
    
    Returns: 
        - train: the training ListDataset
        - test:  the testing ListDataset
    """
    nav_table = __load_raw_nav_table(file_path)
    nav_table = __fill_nav_dataframe(nav_table)
    train_nav_table, test_nav_table = __split_nav_dataframe(
        nav_table, split_date)
    train, test = (
        __convert_to_list_dataset(train_nav_table), 
        __convert_to_list_dataset(test_nav_table)
    )
    return train, test

def split_nav_list_dataset_by_end_dates(nav_dataset, train_end, test_end):
    """
    Extract training and testing dataset from nav_dataset according to 
    the ending date of training and testing dataset. 
    
    Args: 
        - nav_table: (sharable_dataset.SharableListDataset) nav_dataset 
        - train_end: (datetime) end of training 
        - test_end: (datetime) end of testing 
    Returns:
        - train: (ListDataset)
        - test: (ListDataset)
    """
    dataset, _ = Splitter(test_end + timedelta(days=1)).split(nav_dataset)
    train, test = Splitter(train_end + timedelta(days=1)).split(dataset)
    return train, test


def split_nav_dataframe_by_end_dates(nav_table, train_end, test_end):
    """
    Extract training and testing dataset from nav_table according to 
    the ending date of training and testing dataset. 
    
    Args: 
        - nav_table: (pandas.DataFrame) nav table 
        - train_end: (datetime) end of training 
        - test_end: (datetime) end of testing 
    
    Returns:
        - train: (ListDataset)
        - test: (ListDataset)
        
    """
    dataset, _ = __split_nav_dataframe(
        nav_table, test_end
    )
    train, test = __split_nav_dataframe(
        dataset, train_end
    )
    train, test = (
        __convert_to_list_dataset(train), 
        __convert_to_list_dataset(test)
    )
    return train, test

def __split_nav_dataframe(nav_table, split_date):
    """
    Split NAV pandas DataFrame into 
    a training and a testing DataFrame according to a split_date, 
    such that split_date becomes the last date of the 
    training DataFrame.
    
    Args:
        - split_date (datetime.datetime) 
    Returns: 
        - train: the training DataFrame
        - test:  the testing DataFrame
    
    """
    assert split_date in nav_table.index.tolist()
    split_index = nav_table.index.tolist().index(split_date)
    train = nav_table.iloc[:split_index+1]
    test = nav_table.iloc[split_index+1:]
    return train, test






if __name__ == '__main__':
    import os
    nav_files = os.listdir(NAV_DIR)
    file_path = os.path.join(NAV_DIR, nav_files[0])
    print(f"file_path: {file_path}")
    dataset = load_dataset(file_path)   
    from gluonts.model import prophet
    predictor = prophet.ProphetPredictor(
        freq="D", prediction_length=1)
    predictions = predictor.predict(dataset)    
    print('Prediction Result:')
    print(next(predictions))
