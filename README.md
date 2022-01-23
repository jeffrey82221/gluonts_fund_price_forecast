# Introduction
This project aims to predict the future price of mutual fund via GluonTS package.

# TODO: 

- [X] Price Prediction of Single Fund
    - [X] Load fund price csv file into Gluonts time series object. 
    - [X] Process the time series so that the missing prices of holiday can be interpolate.
    - [X] Connect the time series to the Gluonts Model 
    - [ ] Plot the predicted trends (probalistically) from the trained model. 
- [ ] Adapt the Model to the Deep Trainable Models
- [ ] Price Prediction of Multiple Funds 
    - [ ] Load multiple funds and convert to multiple time series 
    - [ ] Parallel loading and processing of multiple time series
    - [ ] Consider multivariate time series Model 
    - [ ] Plot the predicted trends 