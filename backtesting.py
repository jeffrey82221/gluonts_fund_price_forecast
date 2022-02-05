"""
Using Parallel Computing for BackTesting:

NOTE:
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
TODO:
- [X] Modify Implicit Quantile Network:
    - [X] fix cardiinality -> None (default)
    - [X] fix use_feat_dynamic_real -> False (default)
    - [X] fix use_feat_static_cat -> False (default)
    - [X] fix prediction_length and context_length -> prediction_length (default)
    - [X] tune input_size -> 20
"""
import torch
from pts import Trainer
from datetime import datetime
from datetime import timedelta
from sharable_dataset import SharableListDataset
from billiard.pool import Pool
from billiard import cpu_count
from functools import partial
from fund_price_loader import split_nav_list_dataset_by_end_dates
from fund_price_loader import split_nav_dataframe_by_end_dates, load_nav_table, load_dataset
from evaluator import evaluation
import pandas as pd
from gluonts.dataset.util import to_pandas
import matplotlib.pylab as plt
from utils import blockPrinting
import logging
import warnings
warnings.filterwarnings('ignore')

CPU_COUNT = cpu_count()
VERBOSE = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BackTestor:
    def __init__(self, file_path, estimators, prediction_length,
                 eval_period, metric, verbose=VERBOSE, multiprocess=True):
        """
        Args:
            - file_path: the path of the csv storing the nav records of a fund
            - estimators: a dictionary storing the gluonts estimator
            - prediction_length: (number of days) length of each prediction to be evaluate against the ground truth.
            - eval_period:  (number of days) back-testing skipping interval.
            - metric: the metric in agg_metrics to be used for measuring the performance.
            - verbose: whether to monitor the running status
            - multiprocess: whether using multiprocessing to speed up backtesting.
        """
        self.__file_path = file_path
        self.__estimators = estimators
        assert isinstance(self.__estimators, dict)
        self.__prediction_length = prediction_length
        self.__eval_period = eval_period
        self.__metric = metric
        self.__verbose = verbose
        self.__multiprocess = multiprocess
        # for storing dates (at the end of training data) for each iteration of
        # backtesting
        self.__train_ends = None

    def apply(self):
        """
        Apply estimators to parallel_runs
        """
        self.__estimator_performances = dict()
        for estimator_name in self.__estimators:
            print(f"Start Running BackTesting on {estimator_name}")
            begin_time = datetime.now()
            train_ends, performances = self.__parallel_run(
                estimator=self.__estimators[estimator_name])
            if self.__train_ends is not None:
                if self.__verbose:
                    print(f'[apply] {self.__train_ends}')
                    print(f'[apply] {train_ends}')
                assert self.__train_ends == train_ends
            else:
                self.__train_ends = train_ends
            self.__estimator_performances[estimator_name] = performances
            print(
                f"Backtest Time for Estimator {estimator_name}:",
                datetime.now() - begin_time)

    def show_result(self):
        assert self.__train_ends is not None
        assert self.__estimator_performances is not None
        performance_table = pd.DataFrame([self.__train_ends]).T
        performance_table.columns = ['date']
        print('[show_result] create pandas table with train_end dates')
        for estimator_name in self.__estimator_performances:
            assert estimator_name in self.__estimators
            performance_table[estimator_name] = self.__estimator_performances[estimator_name]
            print(f'[show_result] add {estimator_name} results to table')
        performance_table.set_index('date', inplace=True)
        performance_table.plot()
        dataset = load_dataset(self.__file_path)
        to_pandas(list(dataset)[0]).plot(linewidth=2)
        plt.ylabel(self.__metric)
        plt.xlabel('date')
        plt.show()
    # @blockPrinting

    def __parallel_run(self, predictor=None,
                       estimator=None):
        """
        Args:
            - file_path: the path of the csv storing the nav records of a fund
            - duration: (number of days) length of the testing dataset.
            - period:   (number of days) back-testing skipping interval.
            - predictor: the gluonts predictor (for example: facebook prophet)
            - estimator: the pytorchts trainable estimator
        Returns:
            - train_ends: dates at the end of training data for each iteration of backtesting
            - performances: the performance calculated for each iteration of backtesting.
        """
        assert (predictor is not None) or (estimator is not None)
        nav_table = load_nav_table(self.__file_path)
        start_date = nav_table.index.min()
        end_date = nav_table.index.max()

        def get_split_date_gen():
            """
            The generator yeilds:
                - split_date: end date of the testing data.
                - period_end_date: end date of the training data.
            """
            split_date_gen = self.__split_date_generator(
                start_date, end_date, duration=self.__prediction_length, period=self.__eval_period)
            return split_date_gen

        if self.__verbose:
            split_date_gen = get_split_date_gen()
            print(f'Start Date: {start_date}')
            print(f'End Date: {end_date}')
            print(f'Prediction Length: {self.__prediction_length}')
            print(f'Evaluation Period: {self.__eval_period}')
            for train_end, test_end in split_date_gen:
                print(f'Train End: {train_end}; Test End: {test_end}')

        train_ends = list(map(lambda x: x[0], get_split_date_gen()))
        nav_dataset = SharableListDataset(
            nav_table.index[0],
            nav_table.value,
            freq='D'
        )
        if self.__multiprocess:
            with Pool(CPU_COUNT) as p:
                split_date_gen = get_split_date_gen()
                train_test_gen = p.imap(partial(
                    BackTestor.parallel_split,
                    nav_table=nav_dataset
                ), split_date_gen)
                print('[parallel_run] connect to split_date_gen')
                prf_gen = p.imap(partial(
                    BackTestor.parallel_eval,
                    predictor=predictor,
                    estimator=estimator,
                    metric=self.__metric
                ), train_test_gen)
                print('[parallel_run] connect to train_test_gen')
                performances = list(prf_gen)
        else:
            split_date_gen = get_split_date_gen()
            train_test_gen = map(partial(
                BackTestor.parallel_split,
                nav_table=nav_dataset
            ), split_date_gen)
            prf_gen = map(partial(
                BackTestor.parallel_eval,
                predictor=predictor,
                estimator=estimator,
                metric=self.__metric
            ), train_test_gen)
            performances = list(prf_gen)
        return train_ends, performances

    def __split_date_generator(
            self, start_date, end_date, duration=7, period=1):
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

    @staticmethod
    def parallel_split(x, nav_table=None, file_path=None):
        # NOTE: using private method name does not work with p.imap of billiard
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
            print(f'Error in parallel_split: {e}')
            logging.exception(str(e))
            raise e

    @staticmethod
    def parallel_eval(x, predictor=None, estimator=None, metric='MSE'):
        # NOTE: using private method name does not work with p.imap of billiard
        if predictor is None:
            assert estimator is not None
        if estimator is None:
            assert predictor is not None
        train, test = x
        return evaluation(train, test,
                          predictor=predictor,
                          estimator=estimator,
                          verbose=VERBOSE,
                          metric=metric
                          )


if __name__ == '__main__':
    import os
    from fund_price_loader import NAV_DIR
    nav_files = os.listdir(NAV_DIR)
    file_path = os.path.join(NAV_DIR, nav_files[800])
    prediction_length = 14
    eval_period = 70
    mode = 'iq_deep_ar'
    metric = 'MSE'
    assert mode == 'fbprophet' or mode == 'deep_ar' or mode == 'iq_deep_ar'
    """NOTE: fbprophet does not use trainable estimator and does not work with make_evaluation_predictions
    if mode == 'fbprophet':
        from gluonts.model import prophet
        predictor = prophet.ProphetPredictor
    """
    trainer = Trainer(epochs=10, device=DEVICE)
    estimators = dict()
    from pts.model import deepar
    estimator = deepar.DeepAREstimator(
        freq="D",
        prediction_length=prediction_length,
        input_size=17,
        trainer=trainer
    )
    estimators['deep_ar'] = estimator
    print('Create Deep AR Estimator')
    from pts.model import deepar
    from pts.modules.distribution_output import ImplicitQuantileOutput
    estimator = deepar.DeepAREstimator(
        distr_output=ImplicitQuantileOutput(output_domain="Positive"),
        cell_type='GRU',
        input_size=20,  # Tuned
        num_cells=64,
        num_layers=3,
        dropout_rate=0.2,
        embedding_dimension=[4, 4, 4, 4, 16],
        prediction_length=prediction_length,
        context_length=prediction_length,  # Default
        freq='D',
        scaling=True,
        trainer=trainer
    )
    estimators['iq_deep_ar'] = estimator
    print('Create Implict Quantile Deep AR Estimator')
    testor = BackTestor(
        file_path,
        estimators,
        prediction_length,
        eval_period,
        metric,
        verbose=True)
    testor.apply()
    testor.show_result()
