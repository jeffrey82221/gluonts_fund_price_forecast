from sharable_dataset import SharableListDataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
# from math import sqrt
# from sklearn.metrics import mean_squared_error
# import matplotlib.pylab as plt
# from gluonts.dataset.util import to_pandas
from utils import blockPrinting
import warnings
warnings.filterwarnings('ignore')


# @blockPrinting
def evaluation(train, test, predictor=None, estimator=None, verbose=False, metric='MSE'):
    """
    Calculate the performance metrics of a predictor
    on the testing data.

    Args:
        - predictor: Gluonts Predictor - e.g., prophet.ProphetPredictor
        - train: the training ListDataset aka. X
        - test:  the testing ListDataset aka. Y
    Returns:
        - rms: Root Mean Squared Error between prediction and ground truth.
    """
    if isinstance(train, SharableListDataset):
        train = train.to_local()
    if isinstance(test, SharableListDataset):
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
    if verbose: print('[evaluation] predictor created!');
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test,  # test dataset
        predictor=predictor,  # predictor
        num_samples=100,  # number of sample paths we want for evaluation
    )
    if verbose: print('[evaluation] make evaluation!');
    forecasts = list(forecast_it)
    tss = list(ts_it)
    evaluator = Evaluator(num_workers=0)
    agg_metrics, _ = evaluator(
        iter(tss), 
        iter(forecasts), 
        num_series=len(test)
    )
    if verbose:
        print(f'Metrics calculated in Agg_Metrics: {agg_metrics.keys()}')
    return agg_metrics['MSE']
