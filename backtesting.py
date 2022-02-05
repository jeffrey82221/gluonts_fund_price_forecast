"""
Using Parallel Computing for BackTesting:

XXX:
- [X] No Parallel Processing: 11.175463s
- [X] Load Nav Table on Every Process: 08.104964s
- [X] Load Nav Table in the Beginning and Pass to each Process: 05.002175s
- [-] Using Memory Cache to store processed NAV TABLE: 05.435248
- [-] Using pd_share for shared-memory pandas (raising too many error)
- [X] Using list_dataset_share for sharing ListDataset: 04.819891
    - [X] check if the new ListDataset with shared array work (X)
        - [X] Build a Sharable ListDataset
        - [X] Build a Splitter for SharableListDataset
    - [X] convert to sharable list dataset after load_nav_table
    - [X] alter __split_nav_dataframe -> __split_nav_list_dataset such that it can split SharableListDataset
    - [X] split_nav_dataframe_by_end_dates -> split_nav_list_dataset_by_end_dates
    - [X] Testing the execution time of using SharableListDataset
- [X] monitor time for multiprocessing with pytorch trainable models
    - [X] Using multiprocess (30.089s)
    - [X] Using single process (86.263s)
"""
import warnings
warnings.filterwarnings('ignore')
import logging
from utils import blockPrinting
import matplotlib.pylab as plt
from gluonts.dataset.util import to_pandas
import pandas as pd
from evaluator import evaluation
from fund_price_loader import split_nav_dataframe_by_end_dates, load_nav_table, load_dataset
from fund_price_loader import split_nav_list_dataset_by_end_dates
from functools import partial
from billiard import cpu_count
from billiard.pool import Pool
from sharable_dataset import SharableListDataset
from datetime import timedelta
from datetime import datetime
from pts import Trainer
import torch

CPU_COUNT = cpu_count()
VERBOSE = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# @blockPrinting
def parallel_run(file_path, duration, period, predictor=None,
                 estimator=None, metric='MSE', verbose=VERBOSE, multiprocess=True):
    """
    Args:
        - file_path: the path of the csv storing the nav records of a fund
        - duration: (number of days) length of the testing dataset.
        - period:   (number of days) back-testing skipping interval.
        - predictor: the gluonts predictor (for example: facebook prophet)
        - estimator: the pytorchts trainable estimator
    """
    assert (predictor is not None) or (estimator is not None)
    nav_table = load_nav_table(file_path)
    start_date = nav_table.index.min()
    end_date = nav_table.index.max()

    def get_split_date_gen():
        """
        The generator yeilds:
            - split_date: end date of the testing data.
            - period_end_date: end date of the training data.
        """
        split_date_gen = __split_date_generator(
            start_date, end_date, duration=duration, period=period)
        return split_date_gen

    if verbose:
        split_date_gen = get_split_date_gen()
        print(f'Start Date: {start_date}')
        print(f'End Date: {end_date}')
        print(f'Duration: {duration}')
        print(f'Period: {period}')
        for train_end, test_end in split_date_gen:
            print(f'Train End: {train_end}; Test End: {test_end}')

    train_ends = list(map(lambda x: x[0], get_split_date_gen()))
    nav_dataset = SharableListDataset(
        nav_table.index[0],
        nav_table.value,
        freq='D'
    )
    if multiprocess:
        with Pool(CPU_COUNT) as p:
            split_date_gen = get_split_date_gen()
            train_test_gen = p.imap(partial(
                __split,
                nav_table=nav_dataset,
                file_path=None
            ), split_date_gen)
            prf_gen = p.imap(partial(
                __eval,
                predictor=predictor,
                estimator=estimator,
                metric=metric
            ), train_test_gen)
            performances = list(prf_gen)
    else:
        split_date_gen = get_split_date_gen()
        train_test_gen = map(partial(
            __split,
            nav_table=nav_dataset,
            file_path=None
        ), split_date_gen)
        prf_gen = map(partial(
            __eval,
            predictor=predictor,
            estimator=estimator,
            metric=metric
        ), train_test_gen)
        performances = list(prf_gen)
    return train_ends, performances


def show_result(file_path, train_ends, performances, metric):
    performance_table = pd.DataFrame([train_ends, performances]).T
    performance_table.columns = ['date', metric]
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


def __split(x, nav_table=None, file_path=None):
    try:
        train_end, test_end = x
        if nav_table is None:
            assert file_path is not None
            nav_table = load_nav_table(file_path)
            return split_nav_dataframe_by_end_dates(
                nav_table, train_end, test_end)
        else:
            return split_nav_list_dataset_by_end_dates(
                nav_table, train_end, test_end)
    except BaseException as e:
        print(f'Error in __split: {e}')
        logging.exception(str(e))
        raise ValueError('Error in __split')


def __eval(x, predictor=None, estimator=None, metric='MSE'):
    if predictor is None:
        assert estimator is not None
    if estimator is None:
        assert predictor is not None
    train, test = x
    return evaluation(train, test,
                      predictor=predictor,
                      estimator=estimator,
                      verbose=True,
                      metric=metric
                      )


if __name__ == '__main__':
    begin_time = datetime.now()
    import os
    from fund_price_loader import NAV_DIR
    nav_files = os.listdir(NAV_DIR)
    file_path = os.path.join(NAV_DIR, nav_files[800])
    prediction_length = 14
    eval_period = 70
    mode = 'iq_deep_ar'
    metric = 'MSE'
    assert mode == 'fbprophet' or mode == 'deep_ar' or mode == 'iq_deep_ar'
    if mode == 'fbprophet':
        from gluonts.model import prophet
        predictor = prophet.ProphetPredictor
    elif mode == 'deep_ar':
        trainer = Trainer(epochs=10, device=DEVICE)
        from pts.model import deepar
        estimator = deepar.DeepAREstimator(
            freq="D",
            prediction_length=prediction_length,
            input_size=17,
            trainer=trainer
        )
    elif mode == 'iq_deep_ar':
        trainer = Trainer(epochs=10, device=DEVICE)
        from pts.model import deepar
        from pts.modules.distribution_output import ImplicitQuantileOutput
        """
        TODO:
        - [X] fix cardiinality -> None (default)
        - [X] fix use_feat_dynamic_real -> False (default)
        - [X] fix use_feat_static_cat -> False (default)
        - [X] fix prediction_length and context_length -> prediction_length (default)
        - [X] tune input_size -> 20
        """
        estimator = deepar.DeepAREstimator(
            distr_output=ImplicitQuantileOutput(output_domain="Positive"),
            cell_type='GRU',
            input_size=20, # Tuned
            num_cells=64,
            num_layers=3,
            dropout_rate=0.2,
            embedding_dimension = [4, 4, 4, 4, 16],
            prediction_length=prediction_length,
            context_length=prediction_length, # Default
            freq='D',
            scaling=True,
            trainer=trainer
        )

    # Only fbprophet does not use trainable estimator
    if mode == 'fbprophet':
        train_ends, performances = parallel_run(file_path, prediction_length, eval_period,
                           predictor=predictor, metric=metric, multiprocess=True)
    else:
        train_ends, performances = parallel_run(file_path, prediction_length, eval_period,
                           estimator=estimator, metric=metric, multiprocess=True)
    print("Execution Time:", datetime.now() - begin_time)
    show_result(file_path, train_ends, performances, metric)
