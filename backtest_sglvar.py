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
- [X] Refactor so that replication between backtesting and multi_variate_backtesting can be reduced. (see XXX)
    - [X] A abstract BackTestBase object
    - [X] A basic (single variate) BackTest object
    - [X] A multi-variate BackTest object
- [ ] Allow single-variate and multi-variate results to be plot in the same graph.
"""
from backtest_base import BackTestBase
from fund_price_loader import load_nav_table
from nav_splitter import split_nav_list_dataset_by_end_dates
from sharable_dataset import SharableListDataset
from pts import Trainer
import torch
import logging
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SingleVariateBackTestor(BackTestBase):
    """
    Main method of this class:
        run:
            running the backtesting given a nav file path and an estimator
    """

    def __init__(self, file_path, prediction_length,
                 eval_period, metric, verbose=False, multiprocess=True):
        """
        Args:
            - file_path: the path of the csv storing the nav records of a fund
            - prediction_length: (number of days) length of each prediction to be evaluate against the ground truth.
            - eval_period:  (number of days) back-testing skipping interval.
            - metric: the metric in agg_metrics to be used for measuring the performance.
            - verbose: whether to monitor the running status
            - multiprocess: whether using multiprocessing to speed up backtesting.
        """
        super(SingleVariateBackTestor, self).__init__(
            prediction_length,
            eval_period,
            metric,
            verbose=verbose,
            multiprocess=multiprocess
        )
        self.__file_path = file_path

    def load_nav_data(self):
        nav_table = load_nav_table(self.__file_path)
        return nav_table

    def get_start_n_end_dates(self, nav_data):
        start_date = nav_data.index.min()
        end_date = nav_data.index.max()
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
        nav_dataset = SharableListDataset(
            nav_data.index[0],
            nav_data.value,
            freq='D'
        )
        return nav_dataset

    @staticmethod
    def parallel_split(x, nav_data=None):
        try:
            train_end, test_end = x
            return split_nav_list_dataset_by_end_dates(
                nav_data, train_end, test_end)
        except BaseException as e:
            print(f'Error in parallel_split: {e}')
            logging.exception(str(e))
            raise e


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
    from backtest_applier import BackTestApplier
    applier = BackTestApplier(SingleVariateBackTestor,
                              file_path, prediction_length, eval_period, metric,
                              estimators)
    applier.run()
    applier.show_result()
