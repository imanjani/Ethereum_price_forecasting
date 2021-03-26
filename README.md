# Ethereum Price Forecasting with Machine Learning
## An Application of Time Series Regression Models and Neural Networks


## Project Summary
### 1. Data
- Ethereum (ETH) time series: daily close
- Data access: use APIs to get current price information
  - Cryptocurrency data: cryptocompare.com
  - Economic data: fred.stlouisfed.org
  - Cleaning/preprocessing:
    - Convert dates to datetime objects, set datetime index
    - Indices not traded/calculated for weekends/holidays: forward fill previous value until trade activity resumed
    - Address missingness for coins depending on time series start dates

### 2. EDA

- Series contains multiple regime changes
  - Changepoint detection using combination of PELT & FBProphet
  - Size of rolling window set to median regime length (40 days)
- Series is not stationarity
  - First degree differencing produces best results
  - Series is also stationary within rolling forecast window size
- No presence of seasonality component


### 3. Time Series Forecasting
**ARIMA**
- ACF/PACF (Auto & Partial Auto Correlation Functions)


- Autoreggressive: *p < 1*
- Integrated: *d = 1*
- Moving Agerave: *q < 1*
- ARIMA orders modeled (p,d,q): (0,1,0), (1,1,0), (0,1,1)

**LSTM**
- Model configuration
  - Single LSTM layer and a Dense output layer
    - Performance did not improve from configs adding to neural net depth (stacked LSTM layers, dropouts, additional dense layers, etc)
    - Hyperparameters tuned with hyperopt: optimizer, learning rate, units, activation function, vector bias

**Time Series Forecast Results**



- **ARIMA Lowest RMSE: 33.122005** (ARIMA(0,1,0))

(p,d,q)|RMSE
-------|---------
(0,1,0)|33.122005
(1,1,0)|33.853124
(0,1,1)|33.983999

- **LSTM Lowest RMSE: 34.307214** (Adagrad learning rate tuned with TPE algorithm)

Optimizer|RMSE     
---------|---------
RMSprop|34.654522
Adam|34.940433
Adamax|34.568555
Adagrad|34.307214
Adadelta|39.469572

### 4. Exogenous Variables
- The following exogenous data was collected to potentially include as features for price forecasting
  - Ethereum trading data: OHLCV
  - Other Cryptocurrency prices: BTC, XRP, EOS, LTC, XLM, XMR
  - Other Economic Indicators: VIXCLS, TWEXB, EFFR
- Granger causality tests used to determine which to train models with
  - ETH close price Granger caused (one way) by: ETH Volume, XRP, LTC, XMR

**Exogenous Variables Forecast Results**

- **ARIMA(0,1,0) Lowest RMSE: 32.965052** (Improved using ETH Volume)
- **LSTM (Adagrad) Lowest RMSE: 33.605011** (Improved using XRP)
