from datetime import timedelta
from sharable_splitter import Splitter



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
    dataset, _ = split_nav_dataframe(
        nav_table, test_end
    )
    train, test = split_nav_dataframe(
        dataset, train_end
    )
    train, test = (
        __convert_to_list_dataset(train),
        __convert_to_list_dataset(test)
    )
    return train, test


def split_nav_dataframe(nav_table, split_date):
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
    train = nav_table.iloc[:split_index + 1]
    test = nav_table.iloc[split_index + 1:]
    return train, test


