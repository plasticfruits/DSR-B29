#import needed libraries
import numpy as np
import pandas as pd
import pickle


import datetime
from lightgbm import LGBMRegressor     
import category_encoders as ce

def metric(preds, actuals):
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])


def data_load_merge(df):
    #'''
    #This function merges the dataframe, df with the transaction_types and mcc_group.
    #'''    
    store = pd.read_csv('./data/store.csv')
    
    df = pd.merge(df,store, left_on="Store",right_on="Store",how='left',suffixes=('', ''))
    
    return df


def data_clean_prep(df_address):
    #'''
    #cleans and prepares data for model pipeline prediction.
    #'''
    
    df = pd.read_csv(df_address)
    print('The holdout data contains {} data points'.format(df.shape[0]))
    
    df = data_load_merge(df)  #merge df with mcc_group_definition and transaction_types
    
    #drop all rows where the Sales is zero

    df = df[df['Sales'] != 0]
    
    #load mean encode from disk
    mean_encode = pd.read_csv('./data/mean_encode.csv',index_col=0)
    
    #get present year 
    present_year = datetime.datetime.now().year
    
    #change the CompetitionOpenSinceYear from year to age count

    df['CompetitionOpenSinceAge'] = present_year - df['CompetitionOpenSinceYear']
    df['Promo2SinceAge'] = present_year - df['Promo2SinceYear']
    
    #merge mean_encode with X_train and drop Store

    df = pd.merge(df,mean_encode, left_on="Store",right_on="Store",how='left',suffixes=('', ''))
    
    #drop un-used columns 

    df = df.drop(['Date','PromoInterval','CompetitionOpenSinceYear','Promo2SinceYear','Customers','Store'],axis=1)
    
    
    return df