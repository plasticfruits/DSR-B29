# Trees


**Core idea:** break down complex situation (dataset) into s much simpler one.  
**Risks:** they are very sensitive to overfitting (the most of all ML models)  
**Tips:** when modelling I start with LinearRegression or LogisticRegression model and use it as a benchmark for any other model.


## Terminology
* **Root node:** (in the tree) the very top node that represents entire population / sample
* **Splitting:** process of diving a node into 2 or more sub-nodes
* **Decision node:** if sub-node splits into further sub-nodes
* **Pruning:** removing sub-nodes to reduce complexity of tree reducing overfitting
* **Sub-tree / branch:** a sub-section of a tree
* **Pure node:** 
    - if regression problem when adding new data points we make the mean
    - if classification prob,em we take the mod


## Process
How to make the best split so that we get to the pure nodes as fast as possible

1. Choosing feature to split:
    - *Regression problem:* When splitting you take the one with lowest weighted variance reduction of the sub-nodes (combined).
    - *Classification problem:*
        - option 1: **Gini impurity:** we take the sub-nodes with lowest gini impurity - this method is faster than calculating entropy
        - option 2: **entropy:** we calculate the weighted average of entropy fur the sub-nodes and take the feature that gives us the lowest weighted entropy. More computatinal demanding due to log operation.


### Example
Check notebook decision-Trees.ipynb

## Errors
Error = bias + variance + noise
* noise = unmanageable
* variance = fitting to noise
* bias = missing signal

* The Bias / Variance trade-off --> check "../11_Tress/error.md" notebook
    - forests: high-variance, low-bias learners (starting point)
    - boosting: low-variance, high-bias learners (starting point)
    - GridSearch helps identify this


## Bagging
Bootstraping and aggregating (averaging predictors)
Training many individual trees parallelly to use the "power of the crowd"
- each iteration samples the dataset AND (for Rand. Forests) a sample of the features
* Random Forest: bagging and also randomly sampling features of dataset
* Extremely Randomised Trees (ExtraTrees): will not take the best split into account and instead assign a random value for splitting the nodes. 
Benefit: saves computational efficient and performance is almost always comparable.
* Example: see *"../11_Trees/bagging.ipynb"*


## Boosting Methods for Trees
* **AdaBoost** (adaptive boosting): learns from mistakes by increasing weight of missclassified data `AdaBoostClassifier`
    - AdaBoost makes only one split per tree, calculates performance of the tree (aka stump) and assigns more "voting power" by re-assigning weighted averages
    - like Random Forest but instead we take weighted averages
* **Gradient Boosting:** (gradient descent + boosting)  --> Best
We start with the mean as prediction, calculate residuals, fitting the tree and updating residuals, and iterating again
    - Best performer if sample large enough

### Boosting Libraries
* **xgboost:**
    - the famous one
* **lightgbm:**
    - its very very fast! ("my fav")
* **catboost:**
    - the advantage is that it helps to encode the features for you

## Encoding
You should be weary of not having too many (encoded) dummy variables as it will make your features more sparse (many 0's) and hence increase complexity of model and hence more error...

* **One hot encoding:** binary (o's and 1's)
    - ***DANGER!*** when using `pd.get_dummies()`: its not transferable as it might change the assignment of variables. So avoid for production / re-using.
* **Label encoding:** (a: 1, b: 2, c: 3...)
    - very dangerous as there are distances between labels than impact the model
* **Target encoding:** replaces with "posterior probability", that is, the probability in the dataset of that category matching the target value.
* **BaseN encoding:** useful for features with many categories, it reduces the number of dummies for each increase in base=n by making combination of values and cols
    - if base=1 we do a one hot encoding
* **Mean encoding:** use the mean of y for all the records of the feature category

Examples: check *"../11_Trees/encoding.ipynb"* notebook