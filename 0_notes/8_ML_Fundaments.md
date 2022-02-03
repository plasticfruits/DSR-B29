# ML Fundamentals w/ Paul
<br><br>
# **DAY 1**

- Aiming for Strategic Team Lead (Data / Data Science Teams)
- Libraries:
    - scikit --> scikit-learn.org
    - Pandas profiler !!!

* Regression: the target is continuos
* Classification: target is not continuos
* function: "is what producess a car"
* class: "is the factory"

## Linear Regression
### Basic names
* Y (independent Var) = target
* X (dep. var) = features
* The intercept (bias) = ß0
* The slope = ß1
* Accuracy: true positives / true + false positives (???)
* R^2 = how much better is the model than the mean (also a model) // how much of variance explained by features. Problem: R^2 is risky when adding more variables as it does not decrease. 
It ca be negative if it performs worse than the mean (R^2 < 0)
* The parsimonious principle: model sshould be as simple as possible to avoid false correlations / over-fitting
* Kurtosis: how 'high' the pick of the distribution is



SSE is too computational expensive for large n, thats why its common practice to ue gradient descent.

## Bias-Variance trade-off

* Variance = how much does your prediction change if you change the data from the same population
* Over-fitting: if after changing dataset variance increases (model performs worse) // when you start fitting noise you are over-fitting // high-variance is usually a sign of model overfitting the training data
* K-fold cross validation: divide a test set into 5 subsets (default) you rotate 1 subset and use it for testing
* Data leakage: when you use test data for training
* hyperparameter: we asses if use or not through cross-validation (eg. degree of polynomial)
* 




## Norms and Distances

* K-means
* Time-series Data
* Cross-sectional 
* Panel Data
    * If over time reffered as Time-series
* L1 Norm : Manhattan distance - one direction for single step
* L2 Norn: Euclidian distance, combination of directions for single step - usually faster (closer to the origin)

## Regularised regression

* Ridge regression: decreases the beta but never brings it to 0
* Lasso regression does bring beta to 0 .˙. useful for feature selection / reduction. Very good to prevent over-fitting
* optimal value of lamda found through cross-validation (alpha in scikit)

**DS Project Steps**
1. data cleaning
2. imputation
3. encoding
4. data eng. (creating new features/vars)
5. feature selection (Lasso regression)
6. 


## Scaling and Outliers
Scaling a feature makes a huge difference in performance.

* standardise: `StandarsScaler` take away the mean and divide by sd
    - operates on features (i.e. columns)
* scaling: `MinMaxScaler` scale to a given range, typically 0:1
    * its very important to scale the x's for performing gradient-descent so you can compare both beta's and reduce the error
    - bad performance when features have values very far apart (i.e. salaries)
    - operates on features
* normalising: `Normalizer`normalise samples individually to unit form(people often refer to standardise as normalise, but its not the same!)
    - operates on individual samples (i.e. rows)

### Dealing with outliers
* drop
* log-transformation
* will-cox... tranformation
* Tukeys Fences: using IQR
* Mahalanobis Distance: similar to stamdardisation but for multip-dimensional: we compute how far (in SD) a (multi-dimensional) point is from the mean

## Clean Notes After class

### Linear regresion
* Normal equation gives solution, but computational power increases to the power of 3 for each feature

### Errors
* Three type of error when testing model on new dta:
    - bias: high error, underfitting likely
    - variance: high variance can imply overfitting (too sensitive)
    - noise: often irreductible
* Dataset
    - training set: use for training 
    - validation set: for hyper-parameter tunning
    - test set: when the time comes, to make sure the model works "in real life" (use only  time!)

<br><br><br>  

# **DAY 2**


## Gradient Descent

* ways to avoid converging to a local in (instead of global):
    - adam: use the momentum to try to overshoot convergence point
    - start multiple random seeds
    

## Decision Trees
Decision trees split everything "purely" which leads to over-fitting --> "thats why we use random forests"

* **Leaves:** the last nodes of the tree that gives us an indicator
* **Entropy:** Is a measure of uncertainty associated with a given distribution (i.e. if set of 2 var equally split max uncertainty hence max entropy)
* **Gini Impurity:** (default) measures heterogenity of target associated with a given distribution. 
[...] is how often a rand. element would be incorrectly labeled after random labelling
    - You always divide into two groups ONLY - if continuos var we order ascending and split by <= var[0] to var[n]
    - When do we stop? When Gini Impurity of new split is bigger than previous
* pure leave: when Gini Impurty = 0, all entropy is removed.
* **Stem:** after comparing all Gini Impurity across features, we start with the best-performing feature as 1st node

## Random Forests
Good for avoiding over-fitting due to high feature restrictions

* **Bagging Model** (bootstrap aggregation): applied bootstraping and we choose random features
    - Bootstraping: sampling with replacement, helps us restrict the amount of over-fitting we can do
    - OOB (out of bag): observation not taken from bootsraping .˙. can be used for validation later using the OOB error
* **Boosting model:**  

* **feature importance:** (in tree models) how much a feature helped reduce Gini Impurity.
    - You check by shuffling a feature and running it in the random forest, if result worse feature was important

* hyper-prameter tunning: we use `GridSearchCV` library



## Gradient Boosted Trees

* Gradient Boosting: 
    - much more prone to overfit than Random Forest
    - mean_sample_leaf and learning_rate


<br><br><br>  

# **DAY 3**


## Logistic Regression
Used for binary classification on categorical variables
Its a normal linear regression line but under the logistic function so that it tends to +- infinity 

* Maximum likelihood: method for fitting the best line given the available data

## Evaluation Matrices
Particularly important for classification
* **Accuracy:** how often my classifier is right?
    - Plot target always to see if its balanced or imbalanced
    - If **imbalance**, you can use:
        -  **balance accuracy** to put more weight on the other side
        - **oversampling** with **SMOT** (synthetic minority oversampling techniques)
        - only up-sample on train, never on test (nor before doing the train/test split)
* **Precision:** my clasifier says its positive, how often is right?

* binary classifications scenarios (confusion matrix):
    - True  positive
    - True  negative
    - False positive
    - False negative
Often (i.e. cancer) we need to evaluate additionally, we can use precision and recall for this.
* Precision:
* Recall: 
* F1-Score:
* Roc-curve: an evaluation for classification matrices build by applying diff. tresholds 


## Encoding
* sparse model: when we have too many 0's (after encoding too much)
* ONe Hot: (dummy variables) converting categories into k-1 binary cols
* target encoding: make average of targe vals for a category `pip install category encoders`
* Cyclical encoding: used for time (i.e. months 1 comes after 12)
* Multi-Collinearity: correlation between features
* The dummy variable trap: if you make dummies for all features regression model will not be able to distinguish between features

## Feature Selection
1. Uniqueness --> ViF / correlation matrix / ...
2. Relevance --> feature importance / coefficients / p-vals / ...

* Variance inflation factor (ViF): 1 / (1-R2)
    - if ViF > 10 you remove the variable (rule of thumb) as this implies R2 > 0.9
    - process: if multiple, you remove the highest one and run regression again until there is no more ViF. This ensures there is no serious correlation between features.

## K-Means
Types of clustering: https://scikit-learn.org/stable/modules/clustering.html


Unsupervised learning model (that is, no labels)
k = number of clusters
**Saling:** you have to scale all features as you are measuring in euclidian distances
**Use-cases:** mainly user segmentation

* **Centroid:** the average value of all assigned clusters
    - as soon as the centroids don't move any more we have our final centroids
* How many clusters (k)?
    - we stick with the K=n for the n that gives the *smallest variance*
        - too many n's will lead to over-fitting
    - **Elbow-plot** (aka hockey stick): helps to decide on umber of clusters by shows variance reduction
    - **Siluete score:** numeric version of elbow-plot to help decide on the number of clusters


## K-nearest-neighbours
Supervised learning
k = number of neighbours


