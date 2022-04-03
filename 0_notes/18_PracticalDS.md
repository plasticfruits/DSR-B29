
## Probability Calibration

## Libraries
- CalibratedClassifierCV 

* Money at risk: p of returning * basket value

Calibration metrics
* Brier Score: The Brier score measures the mean squared difference between the predicted probability and the actual outcome.
    - The smaller the value the better calibration
    - Calculated on test data

* Calibration Plot: shows the relation between the predicted probability (x-axis) and real probability (y-axis)
    - If no real probability (i.e. binary status of order returned/not returned) you create buckets of observations and their probability, then plot.
        - bucket size is a hyper-parameter that needs to be tested to define
* Re-calibration: adjusts calibration of model (probability score) without affecting model.
    - Model mantains same predictive power



## Dave models for Deployment
* using joblib library from sklearn.

## Deployment

Build micro-service (web service) that contains our model, it takes features in and returns an outcome/probability...
    - IP: instance
    - Service: to process requests / replies
    - Endpoint: what operation we want to perform

* missing values in lvie data?
    - simple imputation: adding mean ? most common values
    - model imputation: using model to impute missing value
## Flask
for running web server

## Model monitoring
Checkin how model performs in production
    - check model BEFORE deploying
    - compare distribution of features and probabilities you want to produce
    - change/point detection (monitoring mean over time)

## Do you really need ML?
If it can be done by set of deterinistic rules, better to do that. Otherwise start with simple model.

