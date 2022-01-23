# first line: 23
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
