from datetime import datetime
import pandas as pd
from src.loader.convertor import convert_to_list_dataset
from src.data_handler.nav_splitter import split_nav_dataframe

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
    dataset = convert_to_list_dataset(nav_table)
    return dataset


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
    raw_nav_table.date = raw_nav_table.date.map(
        lambda x: datetime.strptime(x, '%Y/%m/%d'))
    idx = pd.date_range(
        start=raw_nav_table.date.min(),
        end=raw_nav_table.date.max(),
        freq="D")
    everyday_table = pd.DataFrame(idx, columns=['date'])
    nav_table = everyday_table.merge(raw_nav_table, how='left', on='date')
    nav_table.set_index('date', inplace=True)
    nav_table.interpolate(inplace=True)
    nav_table = nav_table.reset_index().drop_duplicates(
        subset='date').set_index('date')
    assert len(nav_table) == len(set(nav_table.index.tolist()))
    return nav_table


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
    train_nav_table, test_nav_table = split_nav_dataframe(
        nav_table, split_date)
    train, test = (
        convert_to_list_dataset(train_nav_table),
        convert_to_list_dataset(test_nav_table)
    )
    return train, test


if __name__ == '__main__':
    import os
    nav_files = os.listdir(NAV_DIR)
    file_path = os.path.join(NAV_DIR, nav_files[0])
    print(f"file_path: {file_path}")
    dataset = load_dataset(file_path)
    print('dataset:')
    print(dataset)
