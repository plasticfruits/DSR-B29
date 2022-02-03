# Tree for regression prob
#%%
#import libraries
import pandas as pd
import numpy as np

from sklearn.datasets import load_boston
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt


#%%
X,y = load_boston(return_X_y=True)

#train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

#initialize the decisiontreeRegressor
dtc = tree.DecisionTreeRegressor(max_depth=5,random_state=42,criterion='squared_error')
#criterion is the function to measure the quality of a split.

#%%
#fit and return f1_score
dtc.fit(X_train,y_train)

#show decision tree
plt.rcParams["figure.figsize"] = (60,20)
tree.plot_tree(dtc,filled = True);

# give score to model (R2)
dtc.score(X_train,y_train)

# Evaluate score by cross val
cross_val_score(dtc, X, y, scoring="r2")

#%%
# Frank's Code

#load dataset
X,y = load_boston(return_X_y=True)

#train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

dtc = tree.DecisionTreeRegressor(max_depth=5,random_state=42,criterion='mse')

parameters = {
    "max_depth": [2,3,4,5,6,7,8],
    "min_samples_leaf": [1,2,3,4,5]
}

reg = tree.DecisionTreeRegressor(random_state=42)
reg = GridSearchCV(reg, parameters, scoring="explained_variance")
reg.fit(X_train, y_train)

reg.score(X_test, y_test)




###################
# SHOE SIZE TASK
###################

#%% Import & Clean
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

#%%
# import CSV
data = pd.read_csv("./data/shoesize_data/shoesizes.csv", index_col=0)

# filter data
data_clean = data.query("height >= 135 & height <= 200 & shoe_size >= 30 & shoe_size <= 48")
data.describe().transpose()


#%% Fit Model
y = data_clean.loc[:, "shoe_size"]
X = data_clean.drop(columns="shoe_size")

#train,test split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)

#random forest with gini
rf = RandomForestRegressor(criterion='squared_error',n_estimators=150,max_depth=3,n_jobs=-1)

# fit on the data
rf.fit(X_train, y_train)  

# get predictions
y_pred = rf.predict(X_test)

# check MAS for predictions
mean_absolute_error(y_test, y_pred)

# %% Improve with GridSearchCV
from sklearn.model_selection import GridSearchCV

parameter_grid = {
    "n_estimators": [10, 100, 500],
    "max_depth" : [3, 5, 10, 15],
    "min_samples_leaf": [1, 2, 5, 10, 15],
}

# reset model withour params

rf = RandomForestRegressor(n_jobs=-1)

gscv = GridSearchCV(rf, parameter_grid, scoring="neg_mean_absolute_error") # define our goal
gscv.fit(X_train, y_train)

# print best score and estimators
print(gscv.best_estimator_)
print(gscv.best_score_)

# Best estimator:  max_depth=3, min_samples_leaf=2, n_estimators=500


