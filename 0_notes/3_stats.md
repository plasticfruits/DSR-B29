# Stats

* **Python Libraries for stats:**
    - statsmodels (similar to R)
    - scipy (user friendly)

* POC: requirements > data > algo > fit > evaluate >>

### Keywords
* **Data Drift:** means the new data is very different from data used for training model (eg. corona impacted customer behaviour). You can detect it by comparing percentiles of training set with incoming data.
* z-score (or standardization): assigning SD from mean to dataset, where +-1 = 68% and +-2 = 95% ...
* 


### Descriptive Statistics
--> check Map of descriptions!

* **variance:** `sum error^2 / n-1` (of observed data)
* **Mean square error (MSE):** the variance for modelling (of expected data)
* **SD:** `sq. root of variance` (of observed data)
* **Rooted-Mean squared error (RMSE):** is the SD applied to the expected data from a model. Useful to detect models with high outliers! the << the RMSE the better the model.
* **coefficient of variation:** `SD/mean` measures how variable a feature is (very useful metric!!)
    - the lower the Coefvar the better the mean as a descriptor
* **Standard Error (SE):** `SD/sq.root n`
* **95% CI:** `SE*1.96` commonly calculated by bootstraping

* Datasets with long tail: if mean is far away from median the dataset is skewed .˙. mean not good metric

### Parametric vs non-parametric
* **Parametric statistics:** assumes underlying distribution has some defined parameters (i.e. mean and sd for Gaussian dist.)
* **Non-parametric statistics:** no assumption about underlying distribution (not very commonly used...)

* **Parametric models:** model has defined number of parameters (example: linear and logistic regression)
    * you can learn about drivers of business (i.e. what parameter explains how much variability)
* **Non-parametric models:** number of parameters can scale up with amount of data points (example: random forest).
    * lose explainability but gain flexibility (easier to scale model up)
    * Research on AI explainability !!!
    * PCA - Research unsupervised-learning !!!  
* **Mean Absolute Error (MAE):** the average difference between the observations (true values) and model output (predictions). The sign of these differences is ignored so that cancellations between positive and negative values do not occur

### Key distributions
* Poisson: discrete and data not overdispersed (eg. e-commerce)
    * Lambda = the event rate; positive integer Important 
    * property: mean== variance

* Normal (Gaussian): 
    * Properties:
        * Mean = median = mode
        * Unimodal (1 mode only)
        * Symetrical
        * Can be used with z-score (standardized scores)


### Common Pandas functions
* `.describe()` for numeric variables
    * `.describe(include=np.object)` for categorical variables
* `.info()`
* `.unique()` and `.nunique()`
* `.count_values()`
* `.value_counts()`
* `.group_by().agg()`
* `.pd.cut()` and `pd.qcut()` for binning continuous vars into discrete (convert to attribute)
    - common if histogram of variable has multiple modes / not following a specific distribution.


### infering data
* `df.describe().transpose()` for a clear overview of dataset
    * if mean close to 50% data can be normally distributed
    * sd < mean is a good thing
    * if max 1 probably looking at boolean variable (encoded as 0's and 1's)
* `df.describe(percentiles=[0.02, 0.05, .25, .5, .75, .95, .98, 0.99])\.transpose().reset_index(drop=False)` to add custom percentiles to describe function
    * Very strict with model: cut 5% for each side of dataset
    * Strict: cut 2.5% of each side


### SCIPY & Statsmodels
* Probability function: `stats.poisson.pmf()`  or `stats.norm.pmf()`
* Cumulative function: `stats.poisson.cdf()`  or `stats.norm.cdf()` 
* Generate samples: `stats.poisson.rvs(3, size=5000)`   or `stats.norm.rvs()`
* T-test: `stats.ttest_ind(sales_store_a, sales_store_c, equal_var = False)`



### SKLEARN



### Inferential Statistics (i.e. Hypothesis testing)
*inference means generalise*
* Used to compare models
* **p-value:** probability of making a type I error (rejecting H0 when its true)
    * "probability of finding an effect of this size or higher if H0 is true" (the "only" definition)
* **Test-statistic:** measures the effect size on a standard way
* **Effect size (t-value):** distance/magnitude between two averages (for a T-statistic)
* `qqplot()` help us check if data is normally distributed
    * `from statsmodels.graphics.gofplots import qqplot`
* **ANOVA:** how many times is the variance **between** groups larger than the variance **within** groups
* **Coefficient**: is the slop (b) `y = a * bx`
* The H0 of a **single sample** is 0 (no effect)
* **AIC** & **BIC**: error metrics (distance) based on likelihood, with BIC penalising model-complexity more heavily.
* **R^2:** how much % of variance is explained by the variance (the more the better)
    * helpful for describing a model's overall performance but not useful for comparing against models as it does not take complexity into account...
* **F-statistic** (in ANOVA): compares how "better" model performs against mean (also a model) .˙. the higher the better performance.


### Bootstrapping (Central Limit Theorem)
* **Resampling:** action of taking sub-samples of a sample .˙. sample size changes .˙. probability also changes
* **Bootstrapping:** resampling with replacement .˙. probability stays the same
    * we take a sample and calculate the mean iteratively until we achieve a normal distributed sample of means.
    * you can use a **saphiro-wilk** test to check and stop iterating (R) when its normal.

### Effect-Size and Power Tests
AKA analysis of effect size
`from statsmodels.stats import power`
* Power test (1-ß): probability of detecting an effect when there actually is one.
    * the more observations the more power to detect effect across 2 samples
    * `power.tt_ind_solve_power(effect_size=0.5, nobs1=20, alpha=0.05, power=None, ratio=1.0, alternative='two-sided')`
* common effect sizes:
    * correlation coefficient
    * R^2
    * ...
* Cohen's D: most common measure of standarised effect size
    * d represents the distance in the mean (in sd) from samples
    * Small effect size:  d = 0.2
    * Medium effect size: d = 0.5
    * Large effect size:  d = 0.8
* uplift: the increase in conversion for a given x (i.e. day)
* 

