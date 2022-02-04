import logging
from re import S
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions

from pts.model.deepar import DeepAREstimator
from pts.modules.distribution_output import ImplicitQuantileOutput
from pts import Trainer
from pts.dataset.repository.datasets import dataset_recipes

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = get_dataset("m5", regenerate=False)

print(f"Recommended prediction horizon: {dataset.metadata.prediction_length}")
print(f"Frequency of the time series: {dataset.metadata.freq}")
logging.info('Build Trainer')
trainer = Trainer(device=device,
        epochs=20,
        learning_rate=1e-3,
        num_batches_per_epoch=120,
        batch_size=256,
        )
logging.info('Build Estimator')
estimator = DeepAREstimator(
    distr_output=ImplicitQuantileOutput(output_domain="Positive"),
    cell_type='GRU',
    input_size=63,
    num_cells=64,
    num_layers=3,
    dropout_rate=0.2,
    use_feat_dynamic_real=True,
    use_feat_static_cat=True,
    cardinality=[json.loads(cat_feat_info.cardinality) for cat_feat_info in dataset.metadata.feat_static_cat][0],
    embedding_dimension = [4, 4, 4, 4, 16],
    prediction_length=dataset.metadata.prediction_length,
    context_length=dataset.metadata.prediction_length*2,
    freq=dataset.metadata.freq,
    scaling=True,
    trainer=trainer
)
logging.info('Estimate Predictor')
predictor = estimator.train(dataset.train, num_workers=0)
logging.info('Evaluation')

forecast_it, ts_it = make_evaluation_predictions(
    dataset=dataset.test,  # test dataset
    predictor=predictor,  # predictor
    num_samples=100,  # number of sample paths we want for evaluation
)
forecasts = list(forecast_it)
tss = list(ts_it)

evaluator = Evaluator()
agg_metrics, item_metrics = evaluator(
    iter(tss), 
    iter(forecasts), 
    num_series=len(dataset.test)
)

print(json.dumps(agg_metrics, indent=4))

item_metrics.plot(x='MSIS', y='MASE', kind='scatter')
plt.grid(which="both")
plt.show()
# MSIS: Mean Scaled Interval Score
# MASE: Mean Absolute Scaled Error 