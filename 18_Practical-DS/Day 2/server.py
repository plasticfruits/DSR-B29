from flask import Flask, jsonify, request
from joblib import load
import pandas as pd

app = Flask(__name__)
log_reg = load("../models/regression_model_saved.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    basket = request.json["basket"]
    zipCode = request.json["zipCode"]
    totalAmount = request.json["totalAmount"]
    p = probability(basket, zipCode, totalAmount)
    return jsonify({"probability": p}), 201

def extract_basket(basket):
    # extract values for basket
    data = pd.DataFrame(basket, columns=['basket'])
    for i in range(0, 6):
        col_name = f"category_{i}"
        data[col_name] = data["basket"].map(lambda x: x.count(i))

def extract zipcode(zipcode):
    data =
    pd.Categorical(df['zipCode'], categories=list(range(100, 1001)))
    dummies = pd.get_dummies(df.zipCode, prefix='zip')

def probability(basket, zipCode, totalAmount):
    basket = extract_basket(basket)


    p = logreg.predict_proba(in_data)
    print("Processing request: {},{},{}".format(basket, zipCode, totalAmount))

    # load the model from HDD
    # predict with the model

    return p


if __name__ == "__main__":
    app.run(debug=True, port=5001)
