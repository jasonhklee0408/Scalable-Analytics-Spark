#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from dask.distributed import Client
import dask.dataframe as dd
import json

def PA0(user_reviews_csv):
    client = Client()
    client = client.restart()
    
    dtypes = {
    'reviewerID': np.str,
    'asin': np.str,
    'reviewerName': np.str,
    'helpful': np.str,
    'reviewText': np.str,
    'overall': np.float64,
    'summary': np.str,
    'unixReviewTime': np.float64,
    'reviewTime': np.str
    }
    
    df = dd.read_csv(user_reviews_csv, dtype = dtypes)


    df['helpfulVotes'] = df['helpful'].apply(lambda x: int(str(x).split(',')[0].replace('[','')), meta=('helpfulVotes', int))
    df['totalVotes'] = df['helpful'].apply(lambda x: int(str(x).split(', ')[1].replace(']','')), meta=('totalVotes', int))
    df['reviewYear'] = df['reviewTime'].apply(lambda x: int(x[-4:]), meta = ('reviewYear', int))
    df = df[['reviewerID','asin','overall','reviewYear','helpfulVotes', 'totalVotes']]
    
    df_agg = df.groupby('reviewerID').agg({
        'asin':'count',
        'overall':'mean',
        'reviewYear':'min',
        'helpfulVotes':'sum',
        'totalVotes': 'sum'
    }, split_out = 8).reset_index()
    df_agg.columns = ["reviewerID", "number_products_rated", 'avg_ratings', 'reviewing_since', 'helpful_votes', 'total_votes']

    
    submit = df_agg.describe().compute().round(2)    
    with open('results_PA0.json', 'w') as outfile: json.dump(json.loads(submit.to_json()), outfile)

