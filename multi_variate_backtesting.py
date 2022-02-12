"""
Adapt single variate backtesting to multi-variate backtesting

TODO:
- [X] read multiple fund navs.
    - [X] file_path -> file_paths
    - [X] single load_nav_table -> multiple load_nav_table
    - [-] load_dataset -> load_mulvar_dataset
- [X] Split multiple fund navs
    - [X] split multiple SharableListDataset(s)
    - [X] build SharableMultiVariateDataset
- [X] send multivariate estimator to evaluator.py
- [X] add MultiVariate evaluation to evaluator.py
- [X] adapt show_result to multivariate version.
    - [X] allow selection of file (certain fund) for performance visualization
"""
import torch
from pts import Trainer
from datetime import datetime
from datetime import timedelta
from sharable_dataset import SharableListDataset, SharableMultiVariateDataset
from billiard.pool import Pool
# from multiprocessing.dummy import Pool
from billiard import cpu_count
from functools import partial
from nav_splitter import split_nav_list_dataset_by_end_dates
from fund_price_loader import load_nav_table, load_dataset
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
    def __init__(self, file_paths, estimators, prediction_length,
                 eval_period, metric, verbose=VERBOSE, multiprocess=True):
        """
        Args:
            - file_paths: the path of the csv storing the nav records of a fund
            - estimators: a dictionary storing the gluonts estimator
            - prediction_length: (number of days) length of each prediction to be evaluate against the ground truth.
            - eval_period:  (number of days) back-testing skipping interval.
            - metric: the metric in agg_metrics to be used for measuring the performance.
            - verbose: whether to monitor the running status
            - multiprocess: whether using multiprocessing to speed up backtesting.
        """
        self.__file_paths = file_paths
        self.__target_dim = len(file_paths)
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

    def show_result(self, index):
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
        # FIXME: [X] allow selection of file (certain fund) for performance
        # visualization
        dataset = load_dataset(self.__file_paths[index])
        to_pandas(list(dataset)[0]).plot(linewidth=2)
        plt.ylabel(self.__metric)
        plt.xlabel('date')
        plt.show()
    # @blockPrinting

    def __parallel_run(self, predictor=None,
                       estimator=None):
        """
        Args:
            - file_paths: the path of the csv storing the nav records of a fund
            - duration: (number of days) length of the testing dataset.
            - period:   (number of days) back-testing skipping interval.
            - predictor: the gluonts predictor (for example: facebook prophet)
            - estimator: the pytorchts trainable estimator
        Returns:
            - train_ends: dates at the end of training data for each iteration of backtesting
            - performances: the performance calculated for each iteration of backtesting.
        """
        assert (predictor is not None) or (estimator is not None)
        # FIXME: [ ] load_nav_table -> load_mul_nav_tables
        nav_tables = [load_nav_table(file_path)
                      for file_path in self.__file_paths]
        for file, nav_table in zip(self.__file_paths, nav_tables):
            print('File:', file)
            print('Length:', len(nav_table))
        start_date = min(map(lambda x: x.index.min(), nav_tables))
        end_date = max(map(lambda x: x.index.max(), nav_tables))

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
        # FIXME: [X] nav_dataset -> nav_datasets: Create multiple
        # SharableListDataset(s)
        nav_datasets = [SharableListDataset(
            nav_table.index[0],
            nav_table.value,
            freq='D'
        ) for nav_table in nav_tables]
        if self.__multiprocess:
            with Pool(CPU_COUNT) as p:
                split_date_gen = get_split_date_gen()
                train_test_gen = p.imap(partial(
                    # FIXME: [X] split multiple SharableListDataset(s)
                    BackTestor.parallel_split,
                    nav_datasets=nav_datasets
                ), split_date_gen)
                print('[parallel_run] connect to split_date_gen')
                prf_gen = p.imap(partial(
                    BackTestor.parallel_eval,
                    predictor=predictor,
                    estimator=estimator,
                    metric=self.__metric,
                    target_dim=self.__target_dim
                ), train_test_gen)
                print('[parallel_run] connect to train_test_gen')
                performances = list(prf_gen)
        else:
            split_date_gen = get_split_date_gen()
            train_test_gen = map(partial(
                BackTestor.parallel_split,
                nav_datasets=nav_datasets
            ), split_date_gen)
            prf_gen = map(partial(
                BackTestor.parallel_eval,
                predictor=predictor,
                estimator=estimator,
                metric=self.__metric,
                target_dim=self.__target_dim
            ), train_test_gen)
            performances = list(prf_gen)
        del nav_datasets
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
    def parallel_split(x, nav_datasets=None):
        # NOTE: using private method name does not work with p.imap of billiard
        # Before convert those train, test to local ListDataset(s),
        # merge them into SharableMultiVariateDataset
        try:
            train_end, test_end = x
            trains = []
            tests = []
            for nav_dataset in nav_datasets:
                train, test = split_nav_list_dataset_by_end_dates(
                    nav_dataset, train_end, test_end)
                trains.append(train)
                tests.append(test)
            return SharableMultiVariateDataset(
                trains), SharableMultiVariateDataset(tests)
        except BaseException as e:
            print(f'Error in parallel_split: {e}')
            logging.exception(str(e))
            raise e

    @staticmethod
    def parallel_eval(x, predictor=None, estimator=None,
                      metric='MSE', target_dim=1):
        # NOTE: using private method name does not work with p.imap of billiard
        if predictor is None:
            assert estimator is not None
        if estimator is None:
            assert predictor is not None
        train, test = x
        ans = evaluation(train, test,
                         predictor=predictor,
                         estimator=estimator,
                         verbose=VERBOSE,
                         metric=metric,
                         target_dim=target_dim
                         )
        del train, test
        return ans


if __name__ == '__main__':
    import os
    from fund_price_loader import NAV_DIR
    nav_files = os.listdir(NAV_DIR)
    file_paths = [
        os.path.join(
            NAV_DIR, nav_files[800]), os.path.join(
            NAV_DIR, nav_files[801])]
    prediction_length = 14
    eval_period = 70
    mode = 'time_grad'  # TODO: [X] add mode of multivariate models
    metric = 'RMSE'
    assert mode == 'fbprophet' or mode == 'deep_ar' or mode == 'iq_deep_ar' or mode == 'time_grad'
    """NOTE: fbprophet does not use trainable estimator and does not work with make_evaluation_predictions
    if mode == 'fbprophet':
        from gluonts.model import prophet
        predictor = prophet.ProphetPredictor
    """
    trainer = Trainer(epochs=1, device=DEVICE)
    estimators = dict()
    from pts.model.time_grad import TimeGradEstimator
    estimator = TimeGradEstimator(
        target_dim=len(file_paths),
        prediction_length=prediction_length,
        context_length=prediction_length,
        cell_type='GRU',
        input_size=10,
        freq='D',
        loss_type='l2',
        scaling=True,
        diff_steps=100,
        beta_end=0.1,
        beta_schedule="linear",
        trainer=trainer
    )
    estimators['time_grad'] = estimator
    print('Create Deep AR Estimator')
    print('Create Implict Quantile Deep AR Estimator')
    testor = BackTestor(
        file_paths,
        estimators,
        prediction_length,
        eval_period,
        metric,
        verbose=False)
    testor.apply()
    testor.show_result(1)
