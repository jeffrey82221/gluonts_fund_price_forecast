"""
BackTest Applier

- [X] FIXME: 
    - [X] fix the performance aggregation method due to that single-variate train_ends is different for each file. 
    - [X] make sure backtest_sglvar.py / backtest_mulvar.py works
    - [X] allow plotting of all nav_curves (let them seperately plot in another diagram)
"""
import gc
from datetime import datetime
import pandas as pd
from gluonts.dataset.util import to_pandas
import matplotlib.pylab as plt
from matplotlib import gridspec
from fund_price_loader import load_dataset
from backtest_sglvar import SingleVariateBackTestor
from backtest_mulvar import MultiVariateBackTestor

class BackTestApplier:
    """
    Apply backtestor to estimators.
    """

    def __init__(self, testors, estimators, file_paths, prediction_length, eval_period,
                 metric, verbose=False):
        """
        Args:
            - testors: 
                (dict) a dictionary mapping estimator name to BackTestor 
                    (aka. MultiVariateBackTestor/SingleVariateBackTestor)
                (BackTestor) or a BackTestor object to apply to all estimators
            - estimators: a dictionary mapping estimator name to the corresponding Estimator object
            - file_paths: 
                (list) a list of string each indicate the path of a time series csv file
                (str) or a single path for one time series 
            - prediction_length: the days to be forecast
            - eval_period: the duration between evaluation
            - metric: the name of the metric used for evaluation
            - verbose: (bool) whether to print the critical steps.
        """
        assert isinstance(testors, dict)
        self.__testors = testors
        assert isinstance(estimators, dict)
        self.__estimators = estimators
        assert isinstance(file_paths, list)
        self.__file_paths = file_paths
        assert isinstance(prediction_length, int) and prediction_length > 0
        self.__prediction_length = prediction_length
        assert isinstance(eval_period, int) and eval_period > 0
        self.__eval_period = eval_period
        assert isinstance(metric, str)
        self.__metric = metric
        self.__verbose = verbose
    def run(self):
        """
        Run the applier
        """
        self.__estimator_performances = dict()
        self.__estimator_train_ends = dict()
        for estimator_name in self.__estimators:
            print(f"[run] Start Running BackTesting on {estimator_name}")
            begin_time = datetime.now()
            Testor = self.__testors[estimator_name]
            if Testor.__name__ == SingleVariateBackTestor.__name__:
                performances_per_file = dict()
                train_ends_per_file = dict()
                for file_path in self.__file_paths:
                    print(f"[run] Start Running BackTesting on {estimator_name} with {file_path}")
                    backtestor = Testor(
                        file_path,
                        self.__prediction_length,
                        self.__eval_period,
                        self.__metric,
                        verbose=self.__verbose)
                    _train_ends, _performances = backtestor.run(
                        estimator=self.__estimators[estimator_name])
                    assert isinstance(_performances, list)
                    assert isinstance(_train_ends, list)
                    assert len(_performances) == len(_train_ends)
                    train_ends_per_file[file_path] = _train_ends
                    performances_per_file[file_path] = dict(zip(_train_ends, _performances))
                    del _train_ends, _performances
                    gc.collect()
                train_ends, performances = self.__organize_multi_slgvar_results(
                    train_ends_per_file, performances_per_file)
                
            elif Testor.__name__ == MultiVariateBackTestor.__name__:
                backtestor = Testor(
                    self.__file_paths,
                    self.__prediction_length,
                    self.__eval_period,
                    self.__metric,
                    verbose=self.__verbose)
                train_ends, performances = backtestor.run(
                    estimator=self.__estimators[estimator_name])
            else:
                raise ValueError(f'no such BackTestor: {Testor}')
            # XXX: self.__assign_n_check_train_ends(train_ends)
            self.__estimator_performances[estimator_name] = performances
            self.__estimator_train_ends[estimator_name] = train_ends
            print(
                f"[run] Backtest Time for Estimator {estimator_name}:",
                datetime.now() - begin_time)
            del train_ends, performances
            gc.collect()
    # XXX
    def __assign_n_check_train_ends(self, train_ends):
        if self.__train_ends is not None:
            if self.__verbose:
                print(f'[__assign_n_check_train_ends] {self.__train_ends}')
                print(f'[__assign_n_check_train_ends] {train_ends}')
            try:
                assert self.__train_ends == train_ends
            except:
                print(f'[__assign_n_check_train_ends]'
                    f'self.__train_ends[0]:{self.__train_ends[0]}'
                    f'self.__train_ends[-1]:{self.__train_ends[-1]}'
                    f'train_ends[0]:{train_ends[0]}'
                    f'train_ends[-1]:{train_ends[-1]}'
                    )
                raise AssertionError
        else:
            self.__train_ends = train_ends

    def __organize_multi_slgvar_results(self, train_ends_per_file, performances_per_file):
        """
        A method oganizing train_ends and performances from train_ends_per_file and performances_per_file, 
            which are dictionary collecting train_ends and performances from the SingleVariateBackTest run 
            of each nav file. The issue that train_ends are different for each nav files are resolved here.  

        Args:
            - train_ends_per_file: (dict) mapping file_path to train_ends list
            - performances_per_file: (dict) mapping file_path to dictionary mapping 
                train_end dates to performance values
        Returns:
            - train_ends: (list) of train_end dates
            - performances: (list) of performance measured at train_end dates
        """
        __train_ends = [train_end for lt in train_ends_per_file.values() for train_end in lt]
        __train_ends = list(set(__train_ends))
        train_ends = sorted(__train_ends)
        performances = []
        for train_end in train_ends:
            ps = []
            for fp in self.__file_paths:
                if train_end in performances_per_file[fp]:
                    ps.append(performances_per_file[fp][train_end])
            performances.append(sum(ps)/len(ps))
            del ps
            gc.collect()
        return train_ends, performances

    def show_result(self, index=None, normalize_nav=True):
        """
        Args:
            - index: (int) integer for selecting the file to be
                plot against (not None in multivariate setting).
        """
        assert len(self.__estimator_train_ends) > 0
        assert len(self.__estimator_performances) > 0
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 
        ax0 = plt.subplot(gs[0])
        for estimator_name in self.__estimator_performances:
            performance_table = pd.DataFrame([self.__estimator_train_ends[estimator_name]]).T
            performance_table.columns = ['date']
            print('[show_result] create pandas table with train_end dates')
            assert estimator_name in self.__estimators
            performance_table[estimator_name] = self.__estimator_performances[estimator_name]
            print(f'[show_result] add {estimator_name} results to table')
            performance_table.set_index('date', inplace=True)
            performance_table.plot(ax=ax0, label=estimator_name)
        ax0.set_ylabel(self.__metric)
        ax0.legend(
            loc='center left', 
            bbox_to_anchor=(1., 0.5), 
            prop={'size': 3})
        plt.setp(ax0.get_xticklabels(), visible=False)
        ax1 = plt.subplot(gs[1])
        if index is None and len(self.__file_paths) > 1:
            for i in range(len(self.__file_paths)):
                dataset = self.get_visualizing_dataset(index=i)
                dataset_df = to_pandas(list(dataset)[0])
                if normalize_nav:
                    dataset_df = dataset_df/dataset_df.max()
                dataset_df.plot(linewidth=2, ax=ax1, 
                    label=self.__file_paths[i].split('/')[-1])
        else:
            dataset = self.get_visualizing_dataset(index=index)
            dataset_df = to_pandas(list(dataset)[0])
            if normalize_nav:
                dataset_df = dataset_df/dataset_df.max()
            dataset_df.plot(linewidth=2, ax=ax1, 
                label=self.__file_paths[index].split('/')[-1])
        ax1.set_ylabel('normalized nav')
        ax1.legend(
            loc='center left', 
            bbox_to_anchor=(1., 0.5), 
            prop={'size': 3})
        plt.subplots_adjust(hspace=.0)
        plt.xlabel('date')
        plt.show()

    def get_visualizing_dataset(self, index=None):
        """
        Args:
            index: (int) integer for selecting the file to be
                plot against (not None in multivariate setting).
        """
        if isinstance(self.__file_paths, str):
            assert index is None
            dataset = load_dataset(self.__file_paths)
        else:
            assert isinstance(index, int)
            assert index >= 0 and index < len(self.__file_paths)
            dataset = load_dataset(self.__file_paths[index])
        return dataset
