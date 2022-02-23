import os
from datetime import timedelta
from fund_price_loader import load_nav_table
def load_nav_data(file_path):
    nav_table = load_nav_table(file_path)
    return nav_table
def get_start_n_end_dates(nav_data):
    start_date = nav_data.index.min()
    end_date = nav_data.index.max()
    return start_date, end_date
def get_start_n_end_dates_mul(nav_data):
    start_date = min(map(
        lambda nav_table: nav_table.index.min(), nav_data))
    end_date = max(map(
        lambda nav_table: nav_table.index.max(), nav_data))
    return start_date, end_date

def get_split_date_gen(start_date, end_date):
    """
    The generator yeilds:
        - split_date: end date of the testing data.
        - period_end_date: end date of the training data.
    """
    split_date_gen = __split_date_generator(
        start_date, end_date, duration=14, period=70)
    return split_date_gen

def __split_date_generator(start_date, end_date, duration=7, period=1):
    """
    A generator generating the dates splitting the nav_table into
    multiple training and testing for back-testing.

    Args:
        - start_date: start date of the nav_table.
        - end_date:   end date of the nav_table.
        - duration: (number of days) length of the testing dataset.
        - period:   (number of days) back-testing skipping interval.

    Yields:
        - split_date: end date of the testing data.
        - period_end_date: end date of the training data.

    Note: testing dataset have dates after the training data.
    """
    period_end_date = start_date + timedelta(days=2 * duration + 1)
    # Allow the training data to have at least `duration` days so it can support
    # trainable models (e.g., DeepAR).
    while period_end_date <= end_date:
        split_date = period_end_date - timedelta(days=duration)
        yield split_date, period_end_date
        period_end_date = period_end_date + timedelta(days=period)

from fund_price_loader import NAV_DIR
nav_files = os.listdir(NAV_DIR)
file_paths = [
    os.path.join(
        NAV_DIR, nav_files[800]), os.path.join(
        NAV_DIR, nav_files[801])]

nav_table1 = load_nav_data(file_paths[0])
nav_table2 = load_nav_data(file_paths[1])
s1, e1 = get_start_n_end_dates(nav_table1)
print(s1, e1)
s2, e2 = get_start_n_end_dates(nav_table2)
print(s2, e2)
s3, e3 = get_start_n_end_dates_mul([nav_table1, nav_table2])
print(s3, e3)

train_ends_1 = list(get_split_date_gen(s1, e1))
train_ends_2 = list(get_split_date_gen(s2, e2))
train_ends_3 = list(get_split_date_gen(s3, e3))

print(train_ends_1[0], train_ends_1[-1])
print(train_ends_2[0], train_ends_2[-1])
print(train_ends_3[0], train_ends_3[-1])
