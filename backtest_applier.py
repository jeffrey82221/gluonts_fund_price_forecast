"""
BackTest Applier

- [ ] FIXME: 
    - [ ] single-variate train_ends is different for each file. fix the performance aggregation method
    - [ ] allow plotting of all nav_curves (let them seperately plot in another diagram)
"""
from datetime import datetime
import pandas as pd
from gluonts.dataset.util import to_pandas
import matplotlib.pylab as plt
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
        self.__train_ends = None
        self.__verbose = verbose
    def run(self):
        """
        Run the applier
        """
        self.__estimator_performances = dict()
        for estimator_name in self.__estimators:
            print(f"Start Running BackTesting on {estimator_name}")
            begin_time = datetime.now()
            Testor = self.__testors[estimator_name]
            if Testor is SingleVariateBackTestor:
                performances_per_file = dict()
                for file_path in self.__file_paths:
                    print(f"Start Running BackTesting on {estimator_name} with {file_path}")
                    backtestor = Testor(
                        file_path,
                        self.__prediction_length,
                        self.__eval_period,
                        self.__metric,
                        verbose=True)
                    train_ends, _performances = backtestor.run(
                        estimator=self.__estimators[estimator_name])
                    assert isinstance(_performances, list)
                    self.__assign_n_check_train_ends(train_ends)
                    assert len(_performances) == len(train_ends)
                    performances_per_file[file_path] = _performances
                performances = list(
                    map(lambda x: sum(x)/len(x), 
                    zip(performances_per_file.items()))
                )
            elif Testor is MultiVariateBackTestor:
                backtestor = Testor(
                    self.__file_paths,
                    self.__prediction_length,
                    self.__eval_period,
                    self.__metric,
                    verbose=True)
                train_ends, performances = backtestor.run(
                    estimator=self.__estimators[estimator_name])
            else:
                raise ValueError('no such BackTestor')
            self.__assign_n_check_train_ends(train_ends)
            self.__estimator_performances[estimator_name] = performances
            print(
                f"Backtest Time for Estimator {estimator_name}:",
                datetime.now() - begin_time)
    def __assign_n_check_train_ends(self, train_ends):
        if self.__train_ends is not None:
            if self.__verbose:
                print(f'[apply] {self.__train_ends}')
                print(f'[apply] {train_ends}')
            assert self.__train_ends == train_ends
        else:
            self.__train_ends = train_ends
    def show_result(self, index=None):
        """
        Args:
            - index: (int) integer for selecting the file to be
                plot against (not None in multivariate setting).
        """
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
        dataset = self.get_visualizing_dataset(index=index)
        to_pandas(list(dataset)[0]).plot(linewidth=2)
        plt.ylabel(self.__metric)
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
            dataset = load_dataset(self.__file_paths[index])
        return dataset
