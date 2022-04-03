#!/usr/bin/python 
from lightgbm import LGBMRegressor     
import category_encoders as ce

import pandas as pd
import numpy as np
import pickle

from functions import metric,data_clean_prep

address = input("Pls give the address/location of the hold-out set: ")

#run the data_clean and prep function
data = data_clean_prep(df_address = address)


print('running pipeline...')

def run_pipeline(data):
    
    #drop all unneccesary columns
    target = data['Sales']  #set the target column
    data = data.drop(['Sales'],axis=1)
    
    # load the model from disk
    pipeline = pickle.load(open('./pipeline/gb_pipeline.pkl', 'rb'))
    result = pipeline.predict(data)
    score = metric(result,target.to_numpy())
    
    print('The accuracy of the model is {}%'.format(score))
    
    #do a few more visaulizations
    return data,result,target

#run the prediction
data,result,target = run_pipeline(data)