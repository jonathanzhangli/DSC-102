import os
import pyspark.sql.functions as F
import pyspark.sql.types as T
from utilities import SEED
# import any other dependencies you want, but make sure only to use the ones
# availiable on AWS EMR

# ---------------- choose input format, dataframe or rdd ----------------------
INPUT_FORMAT = 'dataframe'  # change to 'rdd' if you wish to use rdd inputs
# -----------------------------------------------------------------------------
if INPUT_FORMAT == 'dataframe':
    import pyspark.ml as M
    import pyspark.sql.functions as F
    import pyspark.sql.types as T
    from pyspark.ml.regression import DecisionTreeRegressor
    from pyspark.ml.evaluation import RegressionEvaluator
if INPUT_FORMAT == 'koalas':
    import databricks.koalas as ks
elif INPUT_FORMAT == 'rdd':
    import pyspark.mllib as M
    from pyspark.mllib.feature import Word2Vec
    from pyspark.mllib.linalg import Vectors
    from pyspark.mllib.linalg.distributed import RowMatrix
    from pyspark.mllib.tree import DecisionTree
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.linalg import DenseVector
    from pyspark.mllib.evaluation import RegressionMetrics


# ---------- Begin definition of helper functions, if you need any ------------

# def task_1_helper():
#   pass

# -----------------------------------------------------------------------------

def task_1(data_io, review_data, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    overall_column = 'overall'
    # Outputs:
    mean_rating_column = 'meanRating'
    count_rating_column = 'countRating'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    joined = product_data.join(review_data, product_data.asin==review_data.asin, how="left")\
                         .select(product_data.asin, review_data.overall)
    
    #filtered_overall = joined.filter(joined[overall_column] is not None)
    
    
    stats = joined.groupBy(asin_column).agg(avg(overall_column).alias("avg_rating"), \
                                            func.count(overall_column).alias("count"))
    
    stats = stats.replace(0, None)
    
    description = stats.describe().collect()
    print(description)


    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    # Calculate the values programmaticly. Do not change the keys and do not
    # hard-code values in the dict. Your submission will be evaluated with
    # different inputs.
    # Modify the values of the following dictionary accordingly.
    res = {
        'count_total': None,
        'mean_meanRating': None,
        'variance_meanRating': None,
        'numNulls_meanRating': None,
        'mean_countRating': None,
        'variance_countRating': None,
        'numNulls_countRating': None
    }
    # Modify res:
    res['count_total'] = int(description[0]["asin"])
    
    res['mean_meanRating'] = float(description[1]["avg_rating"])
    
    res['variance_meanRating'] = float(description[2]["avg_rating"]) ** 2
    
    res['numNulls_meanRating'] = int(description[0]["asin"]) - int(description[0]["avg_rating"])
    
    res['mean_countRating'] = float(description[1]["count"])
    
    res['variance_countRating'] = float(description[2]["count"]) ** 2
    
    res['numNulls_countRating'] = int(description[0]["asin"]) - int(description[0]["count"])



    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_1')
    return res
    # -------------------------------------------------------------------------


def task_2(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    salesRank_column = 'salesRank'
    categories_column = 'categories'
    asin_column = 'asin'
    # Outputs:
    category_column = 'category'
    bestSalesCategory_column = 'bestSalesCategory'
    bestSalesRank_column = 'bestSalesRank'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    
    def first_subcategory(categories):
        if categories is None or len(categories) == 0 or categories[0][0] == "":
            return None
        else:
            return categories[0][0]
            
    udf_first_subcategory = udf(lambda x: first_subcategory(x), StringType())
    
    product_data = product_data.withColumn(category_column, udf_first_subcategory(col(categories_column)))

    exploded = product_data.select(asin_column, explode(salesRank_column))
    
    product_data = product_data.join(exploded, product_data.asin==exploded.asin, how="left")\
                               .select(product_data.asin, product_data.category, exploded.key, exploded.value)
    product_data = product_data.withColumnRenamed("key", bestSalesCategory_column) \
                               .withColumnRenamed("value", bestSalesRank_column)
    
    description = product_data.select(asin_column, \
                                      bestSalesCategory_column, \
                                      bestSalesRank_column, \
                                      category_column).describe().collect()
    

    print(description)
    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_bestSalesRank': None,
        'variance_bestSalesRank': None,
        'numNulls_category': None,
        'countDistinct_category': None,
        'numNulls_bestSalesCategory': None,
        'countDistinct_bestSalesCategory': None
    }
    # Modify res:
    res['count_total'] = int(description[0]["asin"])
    
    res['mean_bestSalesRank'] = float(description[1]["bestSalesRank"])
    
    res['variance_bestSalesRank'] = float(description[2]["bestSalesRank"]) ** 2
    
    res['numNulls_category'] = int(description[0]["asin"]) - int(description[0]["category"])
    
    res['countDistinct_category'] = int(product_data.agg(countDistinct("category")).collect()[0][0])
    
    res['numNulls_bestSalesCategory'] = int(description[0]["asin"]) - int(description[0]["bestSalesCategory"])
    
    res['countDistinct_bestSalesCategory'] = int(product_data.agg(countDistinct(bestSalesCategory_column)).collect()[0][0])

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_2')
    return res
    # -------------------------------------------------------------------------


def task_3(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    price_column = 'price'
    attribute = 'also_viewed'
    related_column = 'related'
    # Outputs:
    meanPriceAlsoViewed_column = 'meanPriceAlsoViewed'
    countAlsoViewed_column = 'countAlsoViewed'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    
    # Filtering data (exploding from double list to single list)
    exploded = product_data.select('asin', explode(related_column))
    filtered = exploded.filter(exploded.key == attribute)
    filtered = filtered.withColumnRenamed("value", related_column)

    # Exploding a second time (From list to individual values)
    exploded_related = filtered.select('asin', explode(related_column))
    exploded_related = exploded_related.withColumnRenamed("col", related_column)\
                                       .withColumnRenamed("asin", "id")

    # Calculate counts
    counts = exploded_related.groupby("id").agg(func.count(related_column).alias(countAlsoViewed_column))
    counts.show()
    
    # Replace 0 with NA
    counts = counts.na.fill(value=0)

    # Calculate means
    prices = product_data.select('asin', 'price')

    with_related_prices = prices.join(exploded_related, prices.asin==exploded_related.related, how="inner")\
                                .select(exploded_related.id, exploded_related.related, prices.price)\
                                .withColumnRenamed("price", "price_related")

    avg_prices = with_related_prices.groupby("id").agg(func.avg("price_related").alias(meanPriceAlsoViewed_column))

    avg_prices.show()

    # Join
    product_data = product_data.select(asin_column)

    product_data = product_data.join(counts, product_data.asin==counts.id, how="left")
    print("First join")
    product_data = product_data.join(avg_prices, product_data.asin==avg_prices.id, how="left")
    print("Second join")

    description = product_data.describe().collect()
    print(description)
    


    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_meanPriceAlsoViewed': None,
        'variance_meanPriceAlsoViewed': None,
        'numNulls_meanPriceAlsoViewed': None,
        'mean_countAlsoViewed': None,
        'variance_countAlsoViewed': None,
        'numNulls_countAlsoViewed': None
    }
    # Modify res:
    
    res["count_total"] = int(description[0][asin_column])
    res["mean_meanPriceAlsoViewed"] = float(description[1][meanPriceAlsoViewed_column])
    res["variance_meanPriceAlsoViewed"] = float(description[2][meanPriceAlsoViewed_column])
    res["numNulls_meanPriceAlsoViewed"] = int(description[0]["asin"]) - int(description[0]["meanPriceAlsoViewed"])
    res["mean_countAlsoViewed"] = float(description[1]["countAlsoViewed"])
    res["variance_countAlsoViewed"] = float(description[2]["countAlsoViewed"])
    res["numNulls_countAlsoViewed"] = int(description[0]["asin"]) - int(description[0]["countAlsoViewed"])

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_3')
    return res
    # -------------------------------------------------------------------------


def task_4(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    price_column = 'price'
    title_column = 'title'
    # Outputs:
    meanImputedPrice_column = 'meanImputedPrice'
    medianImputedPrice_column = 'medianImputedPrice'
    unknownImputedTitle_column = 'unknownImputedTitle'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    # Mean and Median impute
    mean_med = product_data.agg(avg(price_column).alias('mean_price'), \
                                func.expr("percentile(price, 0.5)").alias("median_price"))
    
    mean = mean_med.collect()[0]['mean_price']
    median = mean_med.collect()[0]['median_price']

    product_data = product_data.withColumn(meanImputedPrice_column, col(price_column))
    product_data = product_data.na.fill(value=mean, subset=[meanImputedPrice_column])
    
    product_data = product_data.withColumn(medianImputedPrice_column, col(price_column))
    product_data = product_data.na.fill(value=median, subset=[medianImputedPrice_column])
    
    
    # String column Impute
    product_data = product_data.replace('', None)
    
    description = product_data.select('asin', price_column, title_column, meanImputedPrice_column, \
                                      medianImputedPrice_column).describe().collect()
    
    product_data = product_data.withColumn(unknownImputedTitle_column, col(title_column))
    product_data = product_data.na.fill(value='unknown', subset=[unknownImputedTitle_column])

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_meanImputedPrice': None,
        'variance_meanImputedPrice': None,
        'numNulls_meanImputedPrice': None,
        'mean_medianImputedPrice': None,
        'variance_medianImputedPrice': None,
        'numNulls_medianImputedPrice': None,
        'numUnknowns_unknownImputedTitle': None
    }
    # Modify res:
    res['count_total'] = int(description[0]["asin"])
    
    res['mean_meanImputedPrice'] = float(description[1][meanImputedPrice_column])
    
    res['variance_meanImputedPrice'] = float(description[2][meanImputedPrice_column]) ** 2
    
    res['numNulls_meanImputedPrice'] = int(description[0]["asin"]) - int(description[0][meanImputedPrice_column])
    
    res['mean_medianImputedPrice'] = float(description[1][medianImputedPrice_column])
    
    res['variance_medianImputedPrice'] = float(description[2][medianImputedPrice_column]) ** 2
    
    res['numNulls_medianImputedPrice'] = int(description[0]["asin"]) - int(description[0][medianImputedPrice_column])
    
    res['numUnknowns_unknownImputedTitle'] = int(description[0]["asin"]) - int(description[0][title_column])



    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_4')
    return res
    # -------------------------------------------------------------------------

def task_5(data_io, product_processed_data, word_0, word_1, word_2):
    # -----------------------------Column names--------------------------------
    # Inputs:
    title_column = 'title'
    # Outputs:
    titleArray_column = 'titleArray'
    titleVector_column = 'titleVector'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    def convert_array(title):
        return title.lower().split(' ')
            
    udf_convert_array = udf(lambda x: convert_array(x), ArrayType(StringType()))
    
    product_processed_data = product_processed_data.withColumn(titleArray_column, udf_convert_array(col(title_column)))
    
    product_processed_data.show()
    
    word2Vec = Word2Vec(vectorSize=16, minCount=100, seed=102, numPartitions=4, inputCol=titleArray_column, outputCol=titleVector_column)
    model = word2Vec.fit(product_processed_data)


    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'size_vocabulary': None,
        'word_0_synonyms': [(None, None), ],
        'word_1_synonyms': [(None, None), ],
        'word_2_synonyms': [(None, None), ]
    }
    # Modify res:
    res['count_total'] = product_processed_data.count()
    res['size_vocabulary'] = model.getVectors().count()
    for name, word in zip(
        ['word_0_synonyms', 'word_1_synonyms', 'word_2_synonyms'],
        [word_0, word_1, word_2]
    ):
        res[name] = model.findSynonymsArray(word, 10)
    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_5')
    return res
    # -------------------------------------------------------------------------


def task_6(data_io, product_processed_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    category_column = 'category'
    # Outputs:
    categoryIndex_column = 'categoryIndex'
    categoryOneHot_column = 'categoryOneHot'
    categoryPCA_column = 'categoryPCA'
    # -------------------------------------------------------------------------    

    # ---------------------- Your implementation begins------------------------
    # String Indexer
    indexer = StringIndexer(inputCol=category_column, outputCol=categoryIndex_column)
    indexed = indexer.fit(product_processed_data).transform(product_processed_data)
    
    # One Hot Encodding
    ohe = OneHotEncoderEstimator(inputCols=[categoryIndex_column], outputCols=[categoryOneHot_column], dropLast=False)
    model = ohe.fit(indexed)
    encoded = model.transform(indexed)
    
    # PCA
    pca = PCA(k=15, inputCol=categoryOneHot_column, outputCol=categoryPCA_column)
    pca_model = pca.fit(encoded)
    reduced = pca_model.transform(encoded)
    
    # Summary
    summarizer = Summarizer.metrics('mean')
    
    averages_OHE = reduced.select(summarizer.summary(reduced[categoryOneHot_column])).collect()[0][0][0]
    
    averages_PCA = reduced.select(summarizer.summary(reduced[categoryPCA_column])).collect()[0][0][0]

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'meanVector_categoryOneHot': [None, ],
        'meanVector_categoryPCA': [None, ]
    }
    # Modify res:
    res['count_total'] = int(reduced.count())
    
    res['meanVector_categoryOneHot'] = averages_OHE
    
    res['meanVector_categoryPCA'] = averages_PCA



    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_6')
    return res
    # -------------------------------------------------------------------------

    
def task_7(data_io, train_data, test_data):
    
    # ---------------------- Your implementation begins------------------------

    # Train a DecisionTree model.
    dt = DecisionTreeRegressor(labelCol="overall", featuresCol="features", maxDepth=5)

    # Train model.  This also runs the indexers.
    model = dt.fit(train_data)

    # Make predictions.
    predictions = model.transform(test_data)

    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(labelCol="overall", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    
    # -------------------------------------------------------------------------
    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': None
    }
    # Modify res:
    res['test_rmse'] = rmse


    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_7')
    return res
    # -------------------------------------------------------------------------

    
def task_8(data_io, train_data, test_data):
    
    # ---------------------- Your implementation begins------------------------
    
    # Split the data into training and test sets (30% held out for testing)
    (trainingData, validationData) = train_data.randomSplit([0.75, 0.25])
    
    rmses = []
    min_rmse = 0
    best_model = None

    for depth in [5, 7, 9, 12]:
        
        # Train a DecisionTree model.
        dt = DecisionTreeRegressor(labelCol="overall", featuresCol="features", maxDepth=depth)

        # Train model.  This also runs the indexers.
        model = dt.fit(trainingData)

        # Make predictions.
        predictions = model.transform(validationData)
        
        predictions.show(10)
        
        # Get rmse
        evaluator = RegressionEvaluator(labelCol="overall", predictionCol="prediction", metricName="rmse")
        rmse = evaluator.evaluate(predictions)
        
        min_rmse = min([min_rmse, rmse])
        rmses.append(rmse)
        if rmse == min_rmse:
            best_model = model
    
    best_model_prediction = best_model.transform(test_data)
    
    # Get test rmse
    evaluator = RegressionEvaluator(labelCol="overall", predictionCol="prediction", metricName="rmse")
    test_rmse = evaluator.evaluate(best_model_prediction)
    
    
    # -------------------------------------------------------------------------
    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': None,
        'valid_rmse_depth_5': None,
        'valid_rmse_depth_7': None,
        'valid_rmse_depth_9': None,
        'valid_rmse_depth_12': None,
    }
    # Modify res:
    res['test_rmse'] = test_rmse
    res['valid_rmse_depth_5'] = rmses[0]
    res['valid_rmse_depth_7'] = rmses[1]
    res['valid_rmse_depth_9'] = rmses[2]
    res['valid_rmse_depth_12'] = rmses[3]


    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_8')
    return res
    # -------------------------------------------------------------------------
    