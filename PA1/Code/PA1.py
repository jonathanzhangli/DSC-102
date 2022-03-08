from cProfile import run
from dask.distributed import Client, LocalCluster
import time
import json
import dask.dataframe as dd
import numpy as np
import os
import json
import ast


def PA1(user_reviews_csv,products_csv):
    start = time.time()
    client = Client('127.0.0.1:8786')
    client = client.restart()
    print(client)
    
    dtypes_ur = {
        'reviewerID': np.str,
        'asin': np.str, # product ID
        'reviewerName': np.str,
        'helpful': np.object,
        'reviewText': np.str,
        'overall': np.float64,
        'summary': np.str,
        'unixReviewTime': np.float64,
        'reviewTime': np.str
    }

    dtypes_p = {
        'asin': np.str, # product ID
        'salesRank': np.object,
        'imUrl': np.str,
        'categories': np.object,
        'title': np.str,
        'description': np.str,
        'price': np.float64,
        'related': np.object,
        'brand': np.str
    }
        
    #######################
    user_reviews = dd.read_csv('user_reviews_Release.csv', dtype=dtypes_ur)
    products = dd.read_csv('products_Release.csv', dtype=dtypes_p)  

    # Question 1
    percent_products = (products.isna().sum(axis=0) / products['asin'].shape[0]) * 100
    q1_products = round(percent_products.compute(), 2)
    
    percent_ur = (user_reviews.isna().sum(axis=0) / user_reviews['asin'].shape[0]) * 100
    q1_reviews = round(percent_ur.compute(), 2)
    
    # Question 2
    filter_ur = user_reviews[["asin", "overall"]]
    filter_product = products[["asin", "price"]]
    price_ratings = filter_ur.merge(filter_product, on = 'asin', how='inner')
    price_ratings = price_ratings.dropna()
    price_ratings_corr = price_ratings.corr(method='pearson')
    price_ratings_corr_compute = price_ratings_corr.compute()
    q2 = price_ratings_corr_compute.iloc[0,1]
    
    # Question 3
    stats = products["price"].dropna().describe()
    q3 = round(stats.compute().loc[['mean', 'std', 'min', '50%', 'max']], 2)
    
    # Question 4
    copy = products['categories'].dropna()
    products['super_categories'] = copy.map_partitions(lambda a: a.apply(lambda b: eval(b)[0][0]),meta=('object'))
    
    filtered = products[['asin', 'super_categories']]
    q4 = filtered.groupby('super_categories').agg({'asin': 'count'}, split_out=16).compute()

    # Question 5
    filtered_products_q5 = products[["asin"]]
    filtered_ur_q5 = user_reviews[["asin"]]
    joined = filtered_ur_q5.merge(filtered_products_q5, on = "asin", how = "inner").compute()
    q5 = 0

    joined_length = joined.shape[0]
    fr_length = filtered_ur_q5.shape[0].compute()

    if (joined_length != fr_length):
        q5 = 1
        
    # Question 6
    filtered_asin = products['asin'].dropna().compute()
    filtered_related = products[['related']].dropna().compute()
    seen = set(filtered_asin)

    row = filtered_related.iterrows()
    to_compare = []
    q6 = 0
    for i in range(filtered_related.shape[0]):
        n = ast.literal_eval(next(row)[1]['related'])
        to_compare = list(np.concatenate(list(n.values())).flat)
        
        if any([n in seen for n in to_compare]) == False:
            q6 = 1
            break

    ####################### 
    
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