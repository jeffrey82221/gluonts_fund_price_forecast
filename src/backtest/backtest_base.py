"""
BackTest Base object for backtest_mulvar/sglvar

Here, we use parallel computing for backtesting of different time periods. 

Each time periods has its own starting and ending date. 
"""
import warnings
# from utils import blockPrinting
warnings.filterwarnings('ignore')
import abc
import logging
import pandas as pd
from functools import partial
from billiard import cpu_count
from billiard.pool import Pool
from datetime import timedelta
from src.evaluator import evaluation

CPU_COUNT = cpu_count()
VERBOSE = True


class BackTestBase:
    # XXX: common
    def __init__(self, prediction_length,
                 eval_period, metric, verbose=VERBOSE, multiprocess=True):
        """
        Args:
            - file_path: the path of the csv storing the nav records of a fund
            - estimator: gluonts estimator
            - prediction_length: (number of days) length of each prediction to be evaluate against the ground truth.
            - eval_period:  (number of days) back-testing skipping interval.
            - metric: the metric in agg_metrics to be used for measuring the performance.
            - verbose: whether to monitor the running status
            - multiprocess: whether using multiprocessing to speed up backtesting.
        """
        self.__prediction_length = prediction_length
        self.__eval_period = eval_period
        self.__metric = metric
        self.__verbose = verbose
        self.__multiprocess = multiprocess
        # for storing dates (at the end of training data)
        # for each iteration of backtesting

    @abc.abstractmethod
    def load_nav_data(self):
        """
        Get nav pandas table (or tables) as input for run
        Returns:
            - nav_table or nav_tables (depends on the dimension of input)
        Example:
            nav_table = load_nav_table(self.__file_path)
            return nav_table
        """
        pass

    @abc.abstractmethod
    def get_start_n_end_dates(self, nav_data):
        """
        Get start and end dates of the time series from
        the nav_data
        Args:
            - nav_data: a single nav_table or a list of nav_table(s)
        Returns:
            - start_date
            - end_date
        Examples:
            start_date = nav_data.index.min()
            end_date = nav_data.index.max()
        """
        pass

    @abc.abstractmethod
    def share_nav_data(self, nav_data):
        """
        Converting the nav_table(s) to the sharable counterparts
        (e.g., SharableListDataset)
        Args:
            - nav_data: a single nav_table or a list of nav_table(s)
        Returns:
            - shared_nav_data: a memory shared nav_data
        """
        pass

    def run(self, predictor=None,
            estimator=None):
        """
        Args:
            - predictor: the gluonts predictor (for example: facebook prophet)
            - estimator: the pytorchts trainable estimator
        Returns:
            - train_ends: dates at the end of training data for each iteration of backtesting. 
            - performances: a list of values each indicate a performance calculated for a day in train_ends.
        """
        assert (predictor is not None) or (estimator is not None)
        nav_data = self.load_nav_data()
        if isinstance(nav_data, list):
            assert isinstance(nav_data[0], pd.DataFrame)
            target_dim = len(nav_data)
        else:
            assert isinstance(nav_data, pd.DataFrame)
            target_dim = 1
        start_date, end_date = self.get_start_n_end_dates(nav_data)

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
        nav_data = self.share_nav_data(nav_data)

        if self.__multiprocess:
            with Pool(CPU_COUNT) as p:
                split_date_gen = get_split_date_gen()
                train_test_gen = p.imap(partial(
                    type(self).parallel_split,
                    nav_data=nav_data
                ), split_date_gen)
                print('[run] connect to split_date_gen')
                prf_gen = p.imap(partial(
                    type(self).parallel_eval,
                    predictor=predictor,
                    estimator=estimator,
                    metric=self.__metric,
                    target_dim=target_dim
                ), train_test_gen)
                print('[run] connect to train_test_gen')
                performances = list(prf_gen)
        else:
            split_date_gen = get_split_date_gen()
            train_test_gen = map(partial(
                type(self).parallel_split,
                nav_data=nav_data
            ), split_date_gen)
            prf_gen = map(partial(
                type(self).parallel_eval,
                predictor=predictor,
                estimator=estimator,
                metric=self.__metric,
                target_dim=target_dim
            ), train_test_gen)
            performances = list(prf_gen)
        del nav_data
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
    @abc.abstractmethod
    def parallel_split(x, nav_data=None):
        """
        # NOTE: using private method name does not work with p.imap of billiard
        Args:
            - nav_data: a single nav_table or a list of nav_table(s)
        Returns:
            - train: training dataset
            - test:  testing dataset
        Example:
            try:
                train_end, test_end = x
                return split_nav_list_dataset_by_end_dates(
                    nav_data, train_end, test_end)
            except BaseException as e:
                print(f'Error in parallel_split: {e}')
                logging.exception(str(e))
                raise e
        """
        try:
            pass
        except BaseException as e:
            print(f'Error in parallel_split: {e}')
            logging.exception(str(e))
            raise e

    @staticmethod
    def parallel_eval(x, predictor=None, estimator=None, metric='MSE', target_dim=1):
        # NOTE: using private method name does not work with p.imap of billiard
        if predictor is None:
            assert estimator is not None
        if estimator is None:
            assert predictor is not None
        train, test = x
        metric_value = evaluation(train, test,
                                  predictor=predictor,
                                  estimator=estimator,
                                  verbose=VERBOSE,
                                  metric=metric,
                                  target_dim=target_dim
                                  )
        del train, test
        return metric_value
