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
- [X] Adapt the Model to the Deep Trainable Models
    - [X] using pytorch version models with pytorchts package: https://github.com/zalandoresearch/pytorch-ts
    - [ ] Adapt to all examples in pytorch-ts
        - [ ] Implicit Quantile Network 
        - [ ] Multivariate-Flow
        - [ ] Time-Grad
- [ ] Enhance evaluator.py (adapt the evaluation scheme to more metrices: check implicit_quantile_network.py Line.52-67)
- [ ] Consider MultiVariate Mode for Single Fund Prediction: 
    - [ ] Create different technical curves for each fund
        - [ ] Earning of fund in a time period: e.g., (nav tomorrow - nav today) / nav today. (parameter: time_period)
        - [ ] Standard deviation of earning in a time periods. (parameter: earning_time_period, std_time_period) 
        - [ ] Original NAV curve 
    - [ ] Find data object in gluonts for storing multiple time series
    - [ ] Organize of technical curves of a fund into the multi-timeseries objects offered by gluonts. 
    - [ ] Find Multi-Variate Deep Model and incorporate it into the repo. 
- [ ] Allow comparison of backtesting results of different predictors. 
    - [ ] comparison of overall performance
    - [ ] comparison of performance curves on the same plot 
- [ ] Price Prediction of Multiple Funds
    - [ ] Load multiple funds and convert to multiple time series 
    - [ ] Parallel loading and processing of multiple time series
    - [ ] Consider multivariate time series Model
    - [ ] Plot the predicted trends
- [ ] Consider Binance Dataset 
- [ ] Adoption of Real-time Binance Price 

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
