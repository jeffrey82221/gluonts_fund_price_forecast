# Introduction
This project aims to predict the future price of mutual fund via GluonTS package.

# TODO: 

- [X] Price Prediction of Single Fund
    - [X] Load fund price csv file into Gluonts time series object. 
    - [X] Process the time series so that the missing prices of holiday can be interpolate.
    - [X] Connect the time series to the Gluonts Model. 
    - [X] Plot the predicted trends (probalistically) from the trained model. 
- [X] Enhance Evaluation: 
    - [X] Allow splitting of training, validation, testing time series. 
    - [X] Allow evaluation of prediction using RMSE on testing data. 
    - [X] Allow Backtesting (using Off-the-shielf module of Gluonts). 
        - [X] Parallelize BackTesting
            - [X] split_date generator
            - [X] split dataset -> prediction -> evaluation
        - [-] Try sharing of NAV Table Between Process
- [X] Adapt the Model in backtesting to the Deep Trainable Models
    - [X] using pytorch version models with pytorchts package: https://github.com/zalandoresearch/pytorch-ts
    - [ ] Adapt to all examples in pytorch-ts
        - [X] [Implicit Quantile Network](https://github.com/jeffrey82221/gluonts_fund_price_forecast/commit/bcd759538396c91fc3556900d2f69250fdd7a581)
        - [ ] Multivariate-Flow # Next-Up
        - [X] Time-Grad 
- [X] Enhance the evaluation 
    - [X] adapt the evaluation scheme to more metrices: check implicit_quantile_network.py Line.52-67) (TODO: now-fbprophet is not working) 
    - [X] Allow comparison of different models in a single plot
- [ ] Refactor the current architechture such that adapting to Multi-Variate mode can be easier to follow. 
    - [X] Seperate nav splitting methods from fund_price_loader.py to nav_splitter.py
    - [ ] Refactor so that replication between backtesting and multi_variate_backtesting can be reduced. 
- [X] Consider MultiVariate Mode for Single Fund Prediction:
    - [X] Find data object in gluonts for storing multiple time series (check multivariate_dataset_examples.py)
    - [X] Organize of nav curves of multiple funds into the multi-timeseries objects offered by gluonts. 
        - [X] Read nav curves into multiple SharableListDataset 
        - [X] Using Spliter to obtain multiple train, test SharablesListDataset(s) 
        - [X] [build] a SharableMultiVariateDataset which allow storing of multiple sharable target arrays and allow convertion to 
            grouped_list_dataset (check multivariate_dataset_examples.py for progamming the convertion). 
        - [X] Before convert those train, test to local ListDataset(s), merge them into SharableMultiVariateDataset
        - [X] Convert the multivariate_dataset into local grouped dataset using train_grouper and test_grouper
    - [X] Adapt to Multi-Variate Deep Model (see examples of pytorch-ts) and incorporate it into the repo. 
        - [X] Allow evaluation of Multiple Time Series (see plot and MultivariateEvaluator in Time-Grad-Electricity) 
    - [ ] Create different technical curves for each fund
        - [ ] Earning of fund in a time period: e.g., (nav tomorrow - nav today) / nav today. (parameter: time_period)
        - [ ] Standard deviation of earning in a time periods. (parameter: earning_time_period, std_time_period) 
        - [ ] Original NAV curve 
- [ ] Price Prediction of Multiple Funds
    - [ ] Load multiple funds and convert to multiple time series 
    - [ ] Parallel loading and processing of multiple time series
    - [ ] Consider multivariate time series Model
    - [ ] Plot the predicted trends
- [ ] Consider Binance Dataset
- [ ] Adoption of Real-time Binance Price 
- [ ] Allow parameter tuning for each estimator (with ray.tune)

# Install

## Build from new environment
```
python3 -m virtualenv env
source env/bin/activate
sudo pip install --upgrade pip
pip install -r requirements.txt
```

## Using Pre-installed environment
```
source env/bin/activate
```

# Run 
```
python backtesting.py
```
