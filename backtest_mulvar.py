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
- [X] Inherent BackTestBase
"""
import torch
from pts import Trainer
from sharable_dataset import SharableListDataset, SharableMultiVariateDataset
from billiard import cpu_count
from nav_splitter import split_nav_list_dataset_by_end_dates
from fund_price_loader import load_nav_table
import logging
import warnings
warnings.filterwarnings('ignore')

CPU_COUNT = cpu_count()
VERBOSE = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from backtest_base import BackTestBase

class MultiVariateBackTestor(BackTestBase):
    def __init__(self, file_paths, prediction_length, 
        eval_period, metric, verbose=False, multiprocess=True):
        """
        Args:
            - file_paths: the paths of the csv storing the nav records of a fund
            - prediction_length: (number of days) length of each prediction to be evaluate against the ground truth.
            - eval_period:  (number of days) back-testing skipping interval.
            - metric: the metric in agg_metrics to be used for measuring the performance.
            - verbose: whether to monitor the running status
            - multiprocess: whether using multiprocessing to speed up backtesting.
        """
        super(MultiVariateBackTestor, self).__init__(
            prediction_length,
            eval_period,
            metric,
            verbose=verbose,
            multiprocess=multiprocess
        )
        self.__file_paths = file_paths
        self.__verbose = verbose

    def load_nav_data(self):
        nav_tables = [load_nav_table(file_path)
                      for file_path in self.__file_paths]
        if self.__verbose:
            for file, nav_table in zip(self.__file_paths, nav_tables):
                print('File:', file)
                print('Length:', len(nav_table))
        return nav_tables
    
    def get_start_n_end_dates(self, nav_data):
        start_date = min(map(
            lambda nav_table: nav_table.index.min(), nav_data))
        end_date = max(map(
            lambda nav_table: nav_table.index.max(), nav_data))
        return start_date, end_date

    def share_nav_data(self, nav_data):
        """
        Converting the nav_table(s) to the sharable counterparts
        (e.g., SharableListDataset)
        Args:
            - nav_data: a single nav_table or a list of nav_table(s)
        Returns:
            - shared_nav_data: a memory shared nav_data
        """
        nav_dataset = [SharableListDataset(
            nav_table.index[0],
            nav_table.value,
            freq='D'
        ) for nav_table in nav_data]
        return nav_dataset

    @staticmethod
    def parallel_split(x, nav_data=None):
        # NOTE: using private method name does not work with p.imap of billiard
        # Before convert those train, test to local ListDataset(s),
        # merge them into SharableMultiVariateDataset
        try:
            train_end, test_end = x
            trains = []
            tests = []
            for nav_table in nav_data:
                train, test = split_nav_list_dataset_by_end_dates(
                    nav_table, train_end, test_end)
                trains.append(train)
                tests.append(test)
            return SharableMultiVariateDataset(
                trains), SharableMultiVariateDataset(tests)
        except BaseException as e:
            print(f'Error in parallel_split: {e}')
            logging.exception(str(e))
            raise e

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
    trainer = Trainer(epochs=10, device=DEVICE)
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
    print('Create Time Graph Estimator')
    from backtest_applier import BackTestApplier
    applier = BackTestApplier(MultiVariateBackTestor,
                              file_paths, 
                              prediction_length, 
                              eval_period, 
                              metric,
                              estimators)
    applier.run()
    applier.show_result(0)
