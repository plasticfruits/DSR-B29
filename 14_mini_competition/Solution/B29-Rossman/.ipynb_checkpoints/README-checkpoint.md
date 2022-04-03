### DSR Rossman Mini - Competition

This mini competition is adapted from the Kaggle Rossman challenge.

### Dataset

The dataset is made of two csvs:

```
#  store.csv
['Store', 'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']

#  train.csv
['Date', 'Store', 'DayOfWeek', 'Sales', 'Customers', 'Open', 'Promo','StateHoliday', 'SchoolHoliday']
```

```
Id - an Id that represents a (Store, Date) duple within the test set

Store - a unique Id for each store

Sales - the turnover for any given day (this is what you are predicting)

Customers - the number of customers on a given day

Open - an indicator for whether the store was open: 0 = closed, 1 = open

StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None

SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools

StoreType - differentiates between 4 different store models: a, b, c, d

Assortment - describes an assortment level: a = basic, b = extra, c = extended

CompetitionDistance - distance in meters to the nearest competitor store

CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened

Promo - indicates whether a store is running a promo on that day

Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating

Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2
PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store
```

### Language Used : Python --version: 3.8.11


## Repo Contents

1. data_exploration.ipynb: Contains visualizations and explorations of the dataset
2. model_development.ipynb: Model development and comparisons of different models
3. pipeline.py: Takes the holdout data, transforms it and prints out a RMSPE
4. functions.py: Contains all functions used.


## Approach

1. Exploring All Dataset

2. Data Cleaning

3. Visualizations


4. Pipelines
Here I built 2 Pipelines.
<ol>
<li> Baseline Pipeline using Linear Regression </li>
<li> Pipeline using Gradient Boosted Trees </li>
</ol>

For each pipeline, I encoded the 'transaction_type','explanation','explanation_mcc_group' columns using the BaseNEncoder, while the 'agent','direction' columns were encoded using the OneHot-encoder.


## To Run

1. Read the READ.md file
2. `pip install -r requirements.txt`
3. run the jupyter Notebook (descriptives, model_development etc.)
4. To make new predictions with holdout run `python pipeline.py` in terminal
-- insert holdout data address.


## TO Do

If I had more time I could have

1. Model Improvement
- compare other models like Random Forest, SVM etc.
- grid search or bayesian optimization for best hyper-parameters
- using pre-calculated monthly averages over the categories.
2. More Descriptives and Visualization
- Add more graphs and to see more into the Data at the Data Cleaning Stage.


## Scoring Criteria

The competition is scored based on a composite of predictive accuracy and reproducibility.

## Predictive accuracy

The task is to predict the `Sales` of a given store on a given day.

Submissions are evaluated on the root mean square percentage error (RMSPE):

![](./img/rmspe.png)

```python
def metric(preds, actuals):
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])
```
