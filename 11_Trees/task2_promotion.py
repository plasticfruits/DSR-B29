#%% Libs
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score
#from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
from sklearn.model_selection import GridSearchCV

import datetime


#%% Import Data
data = pd.read_csv("data/promotion/train.csv")
data.describe().transpose()

# drop name col
data = data.drop(columns="EmployeeNo")
profile = ProfileReport(data, title="Pandas Profiling Report", explorative=True)
# Qualification: 4.4% NA's
# Last_performance_score: too many 0's (7.6%) -- are they NA's?


# %% --- TIDY

# As qualification is important, we drop rows with NA's
data_clean = data.dropna(subset=["Qualification"])

# Lets drop the last_performance score
data_clean = data_clean.drop(columns="Last_performance_score")

# replace "More than 5" to 6
data_clean.loc[:, "Marital_Status"] = data_clean.loc[:, "Marital_Status"].replace({"Married": 0, "Single": 1, 'Not_Sure': 1})


# Rename "Not_Sure" to Single in "Marital_status"
data_clean.loc[:, "No_of_previous_employers"] = data_clean.loc[:, "No_of_previous_employers"].replace({"More than 5": 6})

# Format age
todaysDate = pd.to_datetime(datetime.date.today())
data_clean["Age"] = (todaysDate - pd.to_datetime(data_clean["Year_of_birth"], format='%Y'))/np.timedelta64(1,'Y')

# Format years in company
data_clean["Years_in_company"] = (todaysDate - pd.to_datetime(data_clean["Year_of_recruitment"], format='%Y'))/np.timedelta64(1,'Y')


# drop DoB
data_clean = data_clean.drop(columns=["Year_of_birth", "Year_of_recruitment"])

# %% --- ENCODE
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce

# Encode Division as labels in single col
le = LabelEncoder()
data_clean["State_Of_Origin"] = le.fit_transform(data_clean["State_Of_Origin"])
data_clean["Division"] = le.fit_transform(data_clean["Division"])


# Encode Divison as one hot encoding
ce_one = ce.OneHotEncoder(cols=['Foreign_schooled', 
                                'Qualification', 
                                'Gender', "Channel_of_Recruitment",
                                "Past_Disciplinary_Action",
                                "Previous_IntraDepartmental_Movement"]) 

data_encode = ce_one.fit_transform(data_clean)

# %% --- PREP MODEL


y = data_encode.loc[:, "Promoted_or_Not"]
X = data_encode.drop(columns="Promoted_or_Not")

#train,test split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)


#%% --- RANDOM FOREST

#random forest with gini
rfc = RandomForestClassifier(random_state=42)

parameter_grid = {
    "n_estimators": [10, 100, 500],
    "min_samples_leaf": [1, 2, 5, 10],
}

gscv = GridSearchCV(rfc, parameter_grid, scoring="f1") # define our goal
gscv.fit(X_train, y_train)

print(f"best RF f1 score is: \n {gscv.best_score_}")
print(f"best RF estimator is: \n {gscv.best_estimator_}")


# %%
