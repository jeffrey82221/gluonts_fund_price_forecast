from datetime import datetime
import pandas as pd
from gluonts.dataset.util import to_pandas
import matplotlib.pylab as plt
from fund_price_loader import load_dataset


class BackTestApplier:
    """
    Apply backtestor to estimators.
    """

    def __init__(self, BackTestor, file_path, prediction_length, eval_period,
                 metric, estimators, verbose=False):
        self.__estimators = estimators
        self.__metric = metric
        self.__file_path = file_path
        # XXX: self.__file_path: move from __init__ to run
        self.__backtestor = BackTestor(
            self.__file_path,
            prediction_length,
            eval_period,
            self.__metric,
            verbose=True)
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
            train_ends, performances = self.__backtestor.run(
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
        if isinstance(self.__file_path, str):
            assert index is None
            dataset = load_dataset(self.__file_path)
        else:
            assert isinstance(index, int)
            dataset = load_dataset(self.__file_path[index])
        return dataset
