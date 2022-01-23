# Introduction
This project aims to predict the future price of mutual fund via GluonTS package.

# TODO: 

- [X] Price Prediction of Single Fund
    - [X] Load fund price csv file into Gluonts time series object. 
    - [X] Process the time series so that the missing prices of holiday can be interpolate.
    - [X] Connect the time series to the Gluonts Model. 
    - [X] Plot the predicted trends (probalistically) from the trained model. 
- [ ] Enhance Evaluation: 
    - [X] Allow splitting of training, validation, testing time series. 
    - [ ] Allow evaluation of prediction using MSE / MAE on testing data. 
    - [ ] Allow backtesting (using Off-the-shield module of Gluonts). 
- [ ] Adapt the Model to the Deep Trainable Models
- [ ] Consider MultiVariate Mode for Single Fund Prediction: 
    - [ ] Create different technical curves for each fund
        - [ ] Earning of fund in a time period: e.g., (nav tomorrow - nav today) / nav today. (parameter: time_period)
        - [ ] Standard deviation of earning in a time periods. (parameter: earning_time_period, std_time_period) 
        - [ ] Original NAV curve 
- [ ] Price Prediction of Multiple Funds
    - [ ] Load multiple funds and convert to multiple time series 
    - [ ] Parallel loading and processing of multiple time series
    - [ ] Consider multivariate time series Model
    - [ ] Plot the predicted trends