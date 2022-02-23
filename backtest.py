"""
Run BackTesting:

TODO:
- [ ] Allow single-variate and multi-variate results to be plot in the same graph.
    - [X] Seperate BackTestor and apply/show_result -> BackTestor + BackTestApplier
    - [X] Let BackTestApplier allow each estimator to be paired with
             a certain BackTestor (mutli-variate or single-variate)
    - [X] For single variate models, allow them to iterate over multiple time series
"""
import warnings
warnings.filterwarnings('ignore')
from pts import Trainer
import torch
from backtest_applier import BackTestApplier
from backtest_sglvar import SingleVariateBackTestor
from backtest_mulvar import MultiVariateBackTestor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    import os
    from fund_price_loader import NAV_DIR
    nav_files = os.listdir(NAV_DIR)
    # file_path = os.path.join(NAV_DIR, nav_files[800])
    file_paths = [
        os.path.join(
            NAV_DIR, nav_files[800]), os.path.join(
            NAV_DIR, nav_files[801])]
    prediction_length = 14
    eval_period = 70
    metric = 'RMSE'
    trainer = Trainer(epochs=10, device=DEVICE)
    estimators = dict()
    print('Create Deep AR Estimator')
    from pts.model import deepar
    estimator = deepar.DeepAREstimator(
        freq="D",
        prediction_length=prediction_length,
        input_size=17,
        trainer=trainer
    )
    estimators['deep_ar'] = estimator
    print('Create Implict Quantile Deep AR Estimator')
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
    print('Create Time Graph Estimator')
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
    testors = {
        'time_grad': MultiVariateBackTestor,
        'iq_deep_ar': SingleVariateBackTestor,
        'deep_ar': SingleVariateBackTestor
    }

    print('Run BackTesting')
    applier = BackTestApplier(testors, estimators,
                              file_paths, prediction_length, eval_period, metric
                              )
    applier.run()
    applier.show_result(0)
