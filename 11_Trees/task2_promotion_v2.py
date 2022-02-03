# %% Libs
from sklearn.linear_model import LogisticRegression
import category_encoders as ce
import pandas as pd
import numpy as np
import missingno as msno
from sklearn.metrics import f1_score
from sklearn.ensemble import GradientBoostingClassifier

# from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from pandas_profiling import ProfileReport
from sklearn.model_selection import GridSearchCV
import datetime


# %% Import Data
data = pd.read_csv("./11_trees/data/promotion/train.csv")
data.dtypes

data.head()

# Visualise NA's
msno.matrix(data)


# drop name col
data = data.drop(columns="EmployeeNo")
profile = ProfileReport(data, title="Pandas Profiling Report", explorative=True)
profile
# Qualification: 4.4% NA's
# Last_performance_score: too many 0's (7.6%) -- are they NA's?

##################
# %% --- TIDY
##################
data_clean = data

# fill NA's for Qualification
data_clean["Qualification"].fillna("na", inplace=True)

# Rename "More than 5" to 6 in "No_of_previous_employers"
data_clean["No_of_previous_employers"].unique()
data_clean["No_of_previous_employers"] = data_clean["No_of_previous_employers"].replace(
    {"More than 5": 6}
)

# replace "More than 5" to 6 in Marital_Status col
data_clean.loc[:, "Marital_Status"] = data_clean.loc[:, "Marital_Status"].replace(
    {"Married": 0, "Single": 1, "Not_Sure": 1}
)

# get todays date age
todaysDate = pd.to_datetime(datetime.date.today())

# Format age
data_clean["Age"] = (
    todaysDate - pd.to_datetime(data_clean["Year_of_birth"], format="%Y")
) / np.timedelta64(1, "Y")

# Format years in company
data_clean["Years_in_company"] = (
    todaysDate - pd.to_datetime(data_clean["Year_of_recruitment"], format="%Y")
) / np.timedelta64(1, "Y")

# convert No_of_previous_employers to int
data_clean["No_of_previous_employers"] = data_clean["No_of_previous_employers"].astype(
    int
)

# Typoin

# Drop unwanted cols
cols_to_drop = [
    "EmployeeNo",
    "Year_of_birth",
    "Last_performance_score",
    "Year_of_recruitment",
]

data_clean = data_clean.drop(columns=cols_to_drop)


##################
# %% --- ENCODE
##################

# # print
# data_clean.dtypes
# data_clean.head()

# Check "State of Origin"
data_clean["State_Of_Origin"].value_counts()

#  Target encoding for "State of Origin"
ce_te = ce.TargetEncoder(cols=["State_Of_Origin"])

# Column to perform encoding
X = data_clean["State_Of_Origin"]
y = data_clean["Promoted_or_Not"]

# Create an object of the Targetencoder
ce_te.fit(X, y)
data_clean["p_state"] = ce_te.transform(X)

# Check "Division"
data_clean["Division"].value_counts()

# Column to perform encoding
X = data_clean["Division"]
y = data_clean["Promoted_or_Not"]

# Create an object of the Targetencoder
ce_te = ce.TargetEncoder(cols=["Division"])
ce_te.fit(X, y)
data_clean["p_division"] = ce_te.transform(X)


# Check "Division"
data_clean["Channel_of_Recruitment"].value_counts()

# Encode other features as one hot encoding
ce_one = ce.OneHotEncoder(
    drop_invariant=True,
    cols=[
        "Foreign_schooled",
        "Qualification",
        "Gender",
        "Channel_of_Recruitment",
        "Past_Disciplinary_Action",
        "Previous_IntraDepartmental_Movement",
    ],
)

data_encode = ce_one.fit_transform(data_clean)


# regularise Age
X = data_encode[["Age"]]
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X)
data_encode["Age_norm"] = min_max_scaler.fit_transform(X)

# regularise Years_in_company
X = data_encode[["Years_in_company"]]
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X)
data_encode["Years_in_company_norm"] = min_max_scaler.fit_transform(X)

# Drop unwanted cols
colt_to_drop = ["State_Of_Origin", "Division", "Years_in_company", "Age"]
data_encode.drop(columns=colt_to_drop, inplace=True)

data_encode.head()


#######################
# %% --- PREP MODEL
#######################

y = data_encode.loc[:, "Promoted_or_Not"]
X = data_encode.drop(columns="Promoted_or_Not")

# train,test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


#######################
# %% --- RAND. FOREST
#######################

# random forest with gini
rfc = RandomForestClassifier(random_state=42)

parameter_grid = {
    "max_depth": [3, 5, 10, 13],
    # "n_jobs": [-1],
    "n_estimators": [10, 100, 500],
    "min_samples_leaf": [1, 2, 5, 10],
}

gscv = GridSearchCV(rfc, parameter_grid)  # define our goal
gscv.fit(X_train, y_train)


# %% - FIT BEST & TEST

# print(f"best RF f1 score is: \n {gscv.best_score_}")
# print(f"best RF estimator is: \n {gscv.best_estimator_}")

model = RandomForestClassifier(
    max_depth=13, min_samples_leaf=2, n_estimators=10, n_jobs=-1, random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
f1_score(y_test, y_pred, average="micro")
# f1 score is 0.92556


#########################
# %% --- GRADIENT BOOSTING // running 4ever
#########################

X_train["No_of_previous_employers"] = X_train["No_of_previous_employers"].astype(int)

xgb_model = GradientBoostingClassifier(random_state=42)
clf = GridSearchCV(xgb_model, parameter_grid)
clf.fit(X_train, y_train)

# %% - FIT BEST & TEST
print(f"best RF estimator is: \n {clf.best_estimator_}")

clf_model = GradientBoostingClassifier(
    max_depth=13, min_samples_leaf=2, n_estimators=10, random_state=42
)
clf_model.fit(X_train, y_train)
clf_y_pred = clf_model.predict(X_test)
f1_score(y_test, clf_y_pred, average="micro")


#########################
# %% --- LINEAR REGRESSION
#########################

# instantiate the model
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)

lr_y_pred = lr.predict(X_test)
f1_score(y_test, lr_y_pred, average="micro")

#
