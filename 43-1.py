#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from cProfile import run
from dask.distributed import Client, LocalCluster
import pandas as pd
import numpy as np
import dask.dataframe as dd
import re
import statistics
import time
import json

def PA1(user_reviews_csv,products_csv):
    start = time.time()
    client = Client('127.0.0.1:8786')
    client = client.restart()
    print(client)
        
    products = dd.read_csv(products_csv)
    user = dd.read_csv(user_reviews_csv)
    #q1 products
    nullsProduct = products.isna().sum()
    sizeProducts = len(products['asin'])
    nullsPropProduct = (nullsProduct/sizeProducts).compute()*100
    
    #q1 reviews
    nullsUser = user.isna().sum()
    sizeUser = len(user['asin'])
    nullsPropUser = (nullsUser/sizeUser).compute()*100
    
    #q2
    productMerge = products[['asin', 'price']]
    userMerge = user[['asin', 'overall']]
    df = productMerge.merge(userMerge, on='asin')[['price','overall']].dropna()
    correlation = df.corr(method='pearson',split_every = 8).compute()
    
    #q3
    res3 = products['price'].describe().compute()
    res3 = res3[['mean','std','50%','min', 'max']]
    
    #q4
    products['categories'] = products['categories'].astype('str').apply(lambda x: re.sub("[^a-zA-Z\d\s:&,\-\#\;]", '', x).split(',')[0])
    products['categories'] = products['categories'].apply(lambda x: "" if x == "nan" else x)
    products['categories'] = products['categories'].apply(lambda x: "Patio, Lawn & Garden" if x == "Patio" else x)
    products['categories'] = products['categories'].apply(lambda x: "Clothing, Shoes & Jewelry" if x == "Clothing" else x)
    products['categories'] = products['categories'].apply(lambda x: "Children's Music" if x == "Childrens Music" else x)
    products['categories'] = products['categories'].apply(lambda x: "Arts, Crafts & Sewing" if x == "Arts" else x)
    
    
    #q5
    compareProducts = products
    compareProducts = compareProducts[['asin']]
    compareProducts['compare'] = 1
    q5merged = compareProducts.merge(user, on = 'asin', how = 'right')
    res5 = 0 
    if q5merged['compare'].isnull().sum() > 0:
        res5 = 1
    
    #q6
    relatedProducts = products[['related']].dropna()
    relatedProducts['related'] = relatedProducts['related'].apply(lambda x: re.findall('\[.*\]', x))
    relatedProducts['related'] = relatedProducts['related'].apply(lambda x: re.sub("\[|\]|'", '', x[0]).split(','))
    relatedProducts = relatedProducts.explode('related').reset_index(drop=True)
    relatedProducts = relatedProducts.rename(columns={'related':'asin'})
    relatedProducts['compare'] = 1
    mergedProducts = products.merge(relatedProducts, on = 'asin', how = 'right')
    res6 = 0
    if mergedProducts['compare'].isnull().sum() > 0:
        res6 = 1

    q1_reviews = nullsPropUser.round(2)
    q1_products = nullsPropProduct.round(2)
    q2 = correlation['overall'].iloc[0].round(2)
    q3 = res3.round(2)
    q4 = products['categories'].value_counts().compute().round(2)
    q5 = res5 
    q6 = res6
    end = time.time()
    runtime = end-start

    # Write your results to "results_PA1.json" here
    with open('OutputSchema_PA1.json','r') as json_file:
        data = json.load(json_file)
        print(data)

        data['q1']['products'] = json.loads(q1_products.to_json())
        data['q1']['reviews'] = json.loads(q1_reviews.to_json())
        data['q2'] = q2
        data['q3'] = json.loads(q3.to_json())
        data['q4'] = json.loads(q4.to_json())
        data['q5'] = q5
        data['q6'] = q6
    
    # print(data)
    with open('results_PA1.json', 'w') as outfile: json.dump(data, outfile)


    return runtime

