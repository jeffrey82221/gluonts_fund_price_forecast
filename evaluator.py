import warnings
warnings.filterwarnings('ignore')
from utils import blockPrinting

from gluonts.model import prophet
from gluonts.dataset.util import to_pandas

import matplotlib.pylab as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

@blockPrinting
def evaluation(predictor, train, test, verbose=False):
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
    predictor = predictor(
        freq="D", 
        prediction_length=list(test)[0]['target'].shape[0]
    )
    predictions = predictor.predict(train)
    if verbose:
        for entry, forecast in zip(train, predictions):
            to_pandas(entry).plot(linewidth=2)
            forecast.plot(color='g', prediction_intervals=[50.0, 90.0])
        to_pandas(list(test)[0]).plot(color='y', linewidth=2)
        plt.plot()
    y_actual = to_pandas(list(test)[0]).values
    y_predicted = list(predictions)[0].mean
    rms = sqrt(mean_squared_error(y_actual, y_predicted))
    return rms


