import dask.dataframe as dd
import numpy as np
import json
from dask.distributed import Client

# Function to compute all aggregates
def compute_aggregates(partition):
    # Floor dividing unixReviewTime by Seconds In a Year
    siy = 31558149.7632
    partition['reviewing_since'] = partition['reviewing_since'] // siy + 1970
    # Aggregating all 5 rows
    ans = partition.groupby("reviewerID").agg({
        'number_products_rated':'count',
        'avg_ratings': 'mean',
        'reviewing_since':'min',
        'helpful_votes':'sum',
        'total_votes': 'sum'},
        split_out=4)
    return ans

def PA0(user_reviews_csv):
    client = Client()
    client = client.restart()

    #######################
    # List of variable names
    dtypes = {
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

    # Reading data
    df = dd.read_csv(user_reviews_csv, usecols=[
        'reviewerID', 'asin', 'helpful', 'overall', 'unixReviewTime'],
        dtype=dtypes)

    # Adding helpful_votes and total_votes columns
    df["helpful_votes"]=df["helpful"].map_partitions(
        lambda a: a.apply(lambda b:int(b.strip('[]').split(', ')[0])),meta=('object'))
    df["total_votes"]=df["helpful"].map_partitions(
        lambda a: a.apply(lambda b:int(b.strip('[]').split(', ')[1])),meta=('object'))

    # Change Column Names
    df = df.rename(columns={"asin": "number_products_rated", "overall": "avg_ratings", "unixReviewTime": "reviewing_since"})

    #######################

    # Change <YOUR_USERS_DATAFRAME> to the dataframe variable in which you have the final users dataframe result
    submit = compute_aggregates(df).compute().describe().round(2)

    # Writing files
    with open('results_PA0.json', 'w') as outfile: json.dump(json.loads(submit.to_json()), outfile)
