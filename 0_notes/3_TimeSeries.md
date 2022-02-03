# Time series
"For every model that is predicting the future, as soon as people react on the model the result of the model will change." Eg. Predict low sales, marketing decides to do a campaign, sales increase.
Bible: https://otexts.com/fpp2/


## Vintage Time-Series = Univariate and 1-step ahead prediction 
We cannot have missing data in univariate data!! 
Can only predict one step in the future, hence not good usecase for distant predictions (as error of predictions accumulates)
"vintage" time-series analysis like exponential and ARIMA cannot handle multiple seasonality

* **Trend:** average changes over time
* **Seasonality:** a repeated pattern that has a fixed period (variance changes over time)
* **Nested seasonality:** when you have multiple seasonality: eg. christmas sales and weekly seasonality
* **Cycle:** a repeated non-seasonal pattern that depends on an external factor
    Example: school holidays and financial crisis are cycles
* Trend and Seasonality can be:
    * **Additive:** it does not change over time (i.e. its steady)
        - We add the decomposed components to reach the series
    * **Multiplicative:** the magnitude of the phenomena can change over time
        - we multiply the decomposed components to reach the series
* **Decomposition:** the process of "breaking" a time-series into three parts so it can be modelled
    - *Seasonal component:*
        - Modeled with Furier 
    - *Trend:*
        - Modeled with regression models
    - *Randomness* (aka noise / residue): that which cannot be explained by either seasonality or trend
        - Modeled using moving averages
* **Stationarity:** stationary data is data without trends or seasonality. It means that a shift in time doesn’t change the shape of the distribution so the mean, variance and covariance are constant over time.
    - (equiv. to p-value in stats)
    - the variance is measured against the trend
    - Data will never be stationary, but we apply some methods to look stationary so we can model
* **Autocorrelation:** correlation with itself in the past, used to find the period (i.e. length of seasonality) 
    - Plotting this will show us the seasonality, which can be imputed to decompose the data
    - we calculate time steps for each point (n-1, n-2, n-3 ... n-m)
    - we test hypotheis of no correlation between any point and n-i points back for i in 0:m
    - in practice you don't want to use too much data as we are looking for more recent patterns
    - a high correlation at a given point (time) indicates that the whole period is correlated, not just that specific point.
* **Autocorrelation Plot**: Best way to find the period for data by plotting and identifying strongest seasonality


### Dealing with non.stationary data (interview Q. !!!)
* **Differencing / integration:** a method to stationarise data by using the differences between observations
    - `from statsmodels.tsa.statespace.tools import diff`
    - `diff(wine['wine_sales'], k_diff=1)`
    - K_diff: degrees of freedom = n -1 number of points to use. Start with 1 and increase until we reject H0 in ADF test
* **ADF Test:** (Augmented Dickey-Fuller Test) for testing if data is stationary or not


### Functions
* `seasonal_decompose(df['value'], model='additive', period=12).plot()` to decompose and plot
    - Cannot handle NA's
    - period is the seasonality:
        - Daily = 7
        - Monthly = 30
        - yearly = 12
*  `decomposed_series = pd.concat([result.observed, result.seasonal, result.trend, result.resid], axis=1)` to concat all series into DF


### Baseline
The simplest representation of trend, seasonality and autocorrelation
* Mean: takes into account all points with same importance
    - better fit for data with seasonality
* Naive: we only look at latest observation (ignore everything else)
    - better fit for highly trended data
* Exponential smoothing: we take all data but give diff. weights to points to define relevance of a point in respect to distance (time)
    - in between the naive and mean baselines
    - often also used to model (not just as a baseline)
    - alpha: the higher the alpha the more wight on the last observations (closer to naive forecast) / and the lower the alpha the closer to the  mean forecast
    - also used as a smoothing technique
    - `from statsmodels.tsa.holtwinters import ExponentialSmoothing`
    ```
    seasonal_periods = 30
    fitted_model = ExponentialSmoothing(train, trend=None, seasonal='add', seasonal_periods=seasonal_periods).fit()
    fitted_model.summary()
    ```
* Seasonal naive: 

### Autoregressive Models

* **LAG:** the number of steps from observation to use as reference
    - Eg. if LAG = 3 then ARIMA is evaluating model with LAG in [1, 2, 3], and then we evaluate with coefficients their significance
    - The importance of the LAG is given by the coefficient that we calculate with ARIMA
* **ARIMA** (Autoregressive Integrated Moving Average): ARIMA models describe the autocorrelations in the data
    - Autoregressive component (p): linear combination of past values; regression against itself
        - The features are the shifted (LAG) data (t-i steps from last observation)
    - Integrating/diff component (d): helps to stabilise the mean reducing trend and seasonality (i.e. make data stationary)
    - Moving Average component (q): weighted moving average of past error
    _ Variations:
        * **Non-seasonal:** `ARIMA(p,d,q)` 
            - ARMA: if data already stationary as no need for integrating
        * **Seasonal:** `SARIMA(p,d,q)(P,D,Q)m` --> seasonal component is modelled with the non-seasonal components, m is the period
* **SARIMAx:** allows you to add external data, can handle trends + seasonability + regressors
    - `from pmdarima import auto_arima` A gridSearch for ARIMA models



## ML Methods (NOT Vintage)
* Two options:
    - 1. Model with trees by extracting feature from time
    - 2. Sequence models: learns from relationship os data-points and target
        - LSTMs
        - RNN
            - AWS Sagemaker Deepar - "state-of-the-art" for multi-sequence time-series
    - 3. Prophet: "ARIMA on steroids" can deal with multiple seasonality

### Prophet
https://peerj.com/preprints/3190/
* Check `Timeseries_with_prophet.ipynb` workbook for example on BTC
* Tagging: it the most impactful configuration by tagging the past (and future if available) with labels so that the model can learn from it if it repeats or not.
* Regressors: you can add them but you need them for the past AND future