import numpy as np
from sharable_dataset import SharableListDataset, SharableMultiVariateDataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator, MultivariateEvaluator
# from math import sqrt
# from sklearn.metrics import mean_squared_error
# import matplotlib.pylab as plt
# from gluonts.dataset.util import to_pandas
from utils import blockPrinting
import warnings
warnings.filterwarnings('ignore')


# @blockPrinting
def evaluation(train, test, predictor=None, estimator=None,
               verbose=False, metric='MSE', target_dim=1):
    """
    Calculate the performance metrics of a predictor
    on the testing data.

    Args:
        - predictor: Gluonts Predictor - e.g., prophet.ProphetPredictor
        - estimator: Gluonts Estimator - e.g., DeepAREstimator
        - train: the training ListDataset aka. X
        - test:  the testing ListDataset aka. Y
    Returns:
        - rms: Root Mean Squared Error between prediction and ground truth.
    """
    assert isinstance(
        train, SharableListDataset) or isinstance(
        train, SharableMultiVariateDataset)
    assert isinstance(
        test, SharableListDataset) or isinstance(
        test, SharableMultiVariateDataset)
    train = train.to_local()
    test = test.to_local()
    if predictor is not None:
        assert estimator is None
        predictor = predictor(
            freq="D",
            prediction_length=list(test)[0]['target'].shape[0]
        )
    else:
        assert (predictor is None) and (estimator is not None)
        predictor = estimator.train(training_data=train, num_workers=0)
    if verbose:
        print('[evaluation] predictor created!')
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test,  # test dataset
        predictor=predictor,  # predictor
        num_samples=100,  # number of sample paths we want for evaluation
    )
    if verbose:
        print('[evaluation] make evaluation!')
    tss = list(ts_it)
    forecasts = list(forecast_it)
    if target_dim > 1:
        evaluator = MultivariateEvaluator(
            target_agg_funcs={'sum': np.mean},
            num_workers=0
        )
    elif target_dim == 1:
        evaluator = Evaluator(num_workers=0)
    else:
        raise ValueError('target_dim should not be less than 1')
    agg_metrics, _ = evaluator(
        iter(tss),
        iter(forecasts),
        num_series=len(test)
    )
    if verbose:
        print(f'Metrics calculated in Agg_Metrics: {agg_metrics.keys()}')
    return agg_metrics[metric]
