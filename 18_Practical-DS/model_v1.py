# %%
import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


# %% -- IMPORT

dir_path = "./return-data/"
files = os.listdir(dir_path)

dfs = []  # an empty list to store df's
for file in sorted(files):
    dir = dir_path + file
    data = pd.read_json(dir, lines=True)  # read data frame from json file
    data["date"] = file
    dfs.append(data)  # append the data frame to the list

# concatenate list of df's into DF
data_raw = pd.concat(dfs, ignore_index=True)
data_raw["date"] = data_raw["date"].str.replace(".txt", "")
data_raw["date"] = pd.to_datetime(data_raw["date"])

# check distribution of class label
print(data_raw.returnLabel.value_counts())
print("Ratio is:", data_raw[data_raw.returnLabel == 1].shape[0] / data_raw.shape[0])
data_raw.head()

# %% -- FEAUTURE ENGINEERING


def date_expand(df):
    """
    Input: a df with column "date"
    Out: expands multiple datetime features to columns
    """
    df["year"] = df.date.dt.year
    df["month"] = df.date.dt.month
    df["dayofweek"] = df.date.dt.dayofweek
    df["weekofyear"] = df.date.dt.isocalendar().week
    return df


# extract date features
data = date_expand(data_raw)

# # avg. per postcode
# returns_by_zip = data.groupby(["zipCode", "returnLabel"])["returnLabel"].size().reset_index()
# returns_by_zip.filter("returnLabel == 1")
# returns_by_zip[returns_by_zip['returnLabel'] > 0]


# extract values for basket
for i in range(0, 6):
    col_name = f"category_{i}"
    data[col_name] = data["basket"].map(lambda x: x.count(i))


# one-hot encoding
data["zipCode"] = pd.Categorical(data["zipCode"], categories=list(range(100, 1001)))
dummies = pd.get_dummies(data.zipCode, prefix="zip")

# merge
data_model = pd.concat([data, dummies], axis=1)

# drop unwanted cols
data_model = data.drop(["transactionId", "basket", "date"], axis=1)

# %% --- TRAIN / TEST SPLIT

y = data_model.loc[:, ["returnLabel"]]
X = data_model.drop(columns="returnLabel")

# train-test split
train, test = train_test_split(data_model, test_size=0.3, shuffle=False)

# check distribution of both datasets
print(
    "Train data: Label ratio is",
    train[train.returnLabel == 1].shape[0] / train.shape[0],
)
print("Test data: Label ratio is", test[test.returnLabel == 1].shape[0] / test.shape[0])

# split
X_train = train.drop(columns="returnLabel")
y_train = train["returnLabel"]
X_test = test.drop(columns="returnLabel")
y_test = test["returnLabel"]
X_train.shape

# %% --- MODEL TRAINING

# Logistic Regression

lrc = LogisticRegression(max_iter=1000)
lrc = lrc.fit(X, y)

lrc_y = lrc.predict(X)
lrc_score = balanced_accuracy_score(y, lrc_y).round(5)

# GBooster
gbt = GradientBoostingClassifier()
gbt.fit(X_train, y_train)

# %% --- EVALUATE

test_scores_reg = lrc.predict(X_test)
test_scores_gbt = gbt.predict(X_test)

reg_score = accuracy_score(y_test, test_scores_reg)
gbot_score = accuracy_score(y_test, test_scores_gbt)

print(f"Linear Regression Accuracy = {reg_score}")
print(f"Gradient Booster Score = {gbot_score}")

# %% Roc Curve

# %% Brier Score

type(lrc_y)
brie_log = pd.concat([pd.DataFrame(lrc_y, columns=["lrc_predicted"]), y], axis=1)

test_scores = pd.DataFrame(lrc.predict_proba(X))
brier_score = test_scores.join(y)
brier_score = brier_score.rename(columns={0: "false", 1: "true", "returnLabel": "y"})

r_score["probability"] = brier_score.apply(
    lambda row: row["false"] if row["y"] == 0 else row["true"], axis=1
)
