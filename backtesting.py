"""
Using Parallel Computing for BackTesting:

XXX:
- [X] No Parallel Processing: 11.175463s
- [X] Load Nav Table on Every Process: 08.104964s
- [X] Load Nav Table in the Beginning and Pass to each Process: 05.002175s
"""
from utils import blockPrinting
import matplotlib.pylab as plt
from gluonts.dataset.util import to_pandas
import pandas as pd
from evaluator import evaluation
from fund_price_loader import split_nav_dataframe_by_end_dates, load_nav_table, load_dataset
from functools import partial
from billiard import cpu_count
from billiard.pool import Pool
from datetime import timedelta
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

CPU_COUNT = cpu_count()
@blockPrinting
def parallel_run(file_path, predictor, duration, period, verbose=False):
    nav_table = load_nav_table(file_path)
    start_date = nav_table.index.min()
    end_date = nav_table.index.max()
    split_date_gen = __split_date_generator(
        start_date, end_date, duration=duration, period=period)
    train_ends = list(map(lambda x: x[0], split_date_gen))
    with Pool(CPU_COUNT) as p:
        split_date_gen = __split_date_generator(
            start_date, end_date, duration=duration, period=period)
        train_test_gen = p.imap(partial(
            __split,
            nav_table=nav_table,
            file_path=None
        ), split_date_gen)
        rmse_gen = p.imap(partial(
            __eval,
            predictor=predictor
        ), train_test_gen)
        rmses = list(rmse_gen)
    if verbose:
        show_result(file_path, train_ends, rmses)
    return train_ends, rmses


def show_result(file_path, train_ends, rmses):
    performance_table = pd.DataFrame([train_ends, rmses]).T
    performance_table.columns = ['date', 'rmse']
    performance_table.set_index('date', inplace=True)
    dataset = load_dataset(file_path)
    performance_table.plot()
    to_pandas(list(dataset)[0]).plot(linewidth=2)
    plt.show()


def __split_date_generator(start_date, end_date, duration=7, period=1):
    """
    A generator generating the dates splitting the nav_table into
    multiple training and testing for back-testing.

    Args:
        - start_date: start date of the nav_table.
        - end_date:   end date of the nav_table.
        - duration: (number of days) length of the testing dataset.
        - period:   (number of days) back-testing interval.

    Yields:
        - split_date: end date of the testing data.
        - period_end_date: end date of the training data.

    Note: testing dataset have dates after the training data.
    """
    period_end_date = start_date + timedelta(days=duration + 1)
    while period_end_date <= end_date:
        split_date = period_end_date - timedelta(days=duration)
        yield split_date, period_end_date
        period_end_date = period_end_date + timedelta(days=period)


def __split(x, nav_table=None, file_path=None):
    train_end, test_end = x
    if nav_table is None:
        assert file_path is not None
        nav_table = load_nav_table(file_path)
    return split_nav_dataframe_by_end_dates(nav_table, train_end, test_end)


def __eval(x, predictor=None):
    train, test = x
    return evaluation(predictor, train, test, verbose=False)


if __name__ == '__main__':
    begin_time = datetime.now()
    import os
    from fund_price_loader import NAV_DIR
    nav_files = os.listdir(NAV_DIR)
    file_path = os.path.join(NAV_DIR, nav_files[800])
    from gluonts.model import prophet
    ans = parallel_run(file_path, prophet.ProphetPredictor, 14, 70)
    print("Execution Time:", datetime.now() - begin_time)
