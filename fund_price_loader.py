import os
from datetime import datetime
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.model import prophet

NAV_DIR = '../fund_price_crawler/nav'

def load_raw_nav_table(file_path):
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

def fill_nav_dataframe(raw_nav_table):
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
    everyday_nav_table = everyday_table.merge(raw_nav_table, how='left', on='date')
    everyday_nav_table.set_index('date', inplace=True)
    everyday_nav_table.interpolate(inplace=True)
    return everyday_nav_table

def load_gluonts_list_dataset(file_path):
    """
    Load a nav csv and convert it to gluonts ListDataset.

    Args:
        - file_path: the path of the nav csv file
    Returns:
        - data: the ListDataset
    """
    nav_table = load_raw_nav_table(file_path)
    nav_table = fill_nav_dataframe(nav_table)
    data = ListDataset([
        {"start": nav_table.index[0],
        "target": nav_table.value}],freq="D")
    return data

if __name__ == '__main__':
    nav_files = os.listdir(NAV_DIR)
    file_path = os.path.join(NAV_DIR, nav_files[0])
    print(f"file_path: {file_path}")
    dataset = load_gluonts_list_dataset(file_path)    
    predictor = prophet.ProphetPredictor(
        freq="D", prediction_length=1)
    predictions = predictor.predict(dataset)    
    print('Prediction Result:')
    print(next(predictions))
