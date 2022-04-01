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
    review_mean = review_data[['asin','overall']].groupby('asin').agg(F.mean('overall'),F.count('overall'))
    joined = product_data.join(review_mean, on='asin', how='left')
    count_total = joined.count()
    mean_meanRating = joined.select(F.avg(F.col('avg(overall)'))).head()[0]
    variance_meanRating = joined.select(F.variance(F.col('avg(overall)'))).head()[0]
    numNulls_meanRating = joined.filter(F.col('avg(overall)').isNull()).count()
    mean_countRating = joined.select(F.avg(F.col('count(overall)'))).head()[0]
    variance_countRating = joined.select(F.variance(F.col('count(overall)'))).head()[0]
    numNulls_countRating = joined.filter(F.col('count(overall)').isNull()).count()




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

    res['count_total'] = count_total
    res['mean_meanRating'] = mean_meanRating
    res['variance_meanRating'] = variance_meanRating
    res['numNulls_meanRating'] = numNulls_meanRating
    res['mean_countRating'] = mean_countRating
    res['variance_countRating'] = variance_countRating
    res['numNulls_countRating'] = numNulls_countRating


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
    product_data_cat = product_data.withColumn('category', F.when((F.size(F.col('categories')[0]) <= 0) | (F.col('categories')[0][0] == ''), None).otherwise(F.col('categories')[0][0]))

    
    key = product_data.select(F.explode(F.col("salesRank")))[['key']]
    value = product_data.select(F.explode(F.col("salesRank")))[['value']]




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

    
    res['count_total'] = product_data.count()
    res['mean_bestSalesRank'] = value.select(F.avg(F.col('value'))).head()[0]
    res['variance_bestSalesRank'] = value.select(F.variance(F.col('value'))).head()[0]
    res['numNulls_category']= product_data_cat.filter(F.col('category').isNull()).count()
    res['countDistinct_category'] = product_data_cat[['category']].distinct().count() - 1
    res['numNulls_bestSalesCategory'] = product_data.filter(F.col("salesRank").isNull()).count() + product_data.filter(F.size(F.col("salesRank")) ==0).count()
    res['countDistinct_bestSalesCategory'] = key.distinct().count()


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
    y = product_data.select('asin',F.explode('related'))
    y = y.filter(F.col('key').contains('also_viewed'))
    y = y.withColumn(countAlsoViewed_column, F.size('value'))
    mean_table = y.select('asin',F.explode('value'))
    join_table = product_data.withColumnRenamed('asin','col')['col','price']
    mean_table = mean_table.join(join_table, on = 'col', how = 'left')
    mean_table = mean_table[['asin','price']].groupby('asin').agg(F.mean('price'))['asin','avg(price)']
    y = y.join(mean_table, on = 'asin', how = 'left')
    total = product_data.join(y, on = 'asin', how = 'left')
    
    count_total = total.count()
    mean_meanPriceAlsoViewed = total.select(F.avg(F.col('avg(price)'))).head()[0]
    variance_meanPriceAlsoViewed = total.select(F.variance(F.col('avg(price)'))).head()[0]
    numNulls_meanPriceAlsoViewed = total.filter(F.col('avg(price)').isNull()).count()
    mean_countAlsoViewed = total.select(F.avg(F.col(countAlsoViewed_column))).head()[0]
    variance_countAlsoViewed = total.select(F.variance(F.col(countAlsoViewed_column))).head()[0]
    numNulls_countAlsoViewed = total.filter(F.col(countAlsoViewed_column).isNull()).count()






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
    res = {
        'count_total': count_total,
        'mean_meanPriceAlsoViewed': mean_meanPriceAlsoViewed,
        'variance_meanPriceAlsoViewed': variance_meanPriceAlsoViewed,
        'numNulls_meanPriceAlsoViewed': numNulls_meanPriceAlsoViewed,
        'mean_countAlsoViewed': mean_countAlsoViewed,
        'variance_countAlsoViewed': variance_countAlsoViewed,
        'numNulls_countAlsoViewed': numNulls_countAlsoViewed

    }



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

    product_data = product_data.withColumn(medianImputedPrice_column, F.col('price'))
    product_data = product_data.withColumn(meanImputedPrice_column, F.col('price'))
    product_data = product_data.withColumn(unknownImputedTitle_column, F.col('title'))
    
    
    
    mean_price = product_data.select(F.avg(F.col('price'))).head()[0]
    median_price = product_data.approxQuantile(medianImputedPrice_column, [0.5],0)[0]
    
    product_data = product_data.na.fill({meanImputedPrice_column:mean_price, medianImputedPrice_column:median_price,unknownImputedTitle_column:'unknown'})
    



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
    res['count_total'] = product_data.count()
    res['mean_meanImputedPrice'] = product_data.select(F.avg(F.col(meanImputedPrice_column))).head()[0]
    res['variance_meanImputedPrice'] = product_data.select(F.variance(F.col(meanImputedPrice_column))).head()[0]
    res['numNulls_meanImputedPrice'] = product_data.filter(F.col(meanImputedPrice_column).isNull()).count()
    res['mean_medianImputedPrice'] = product_data.select(F.avg(F.col(medianImputedPrice_column))).head()[0]
    res['variance_medianImputedPrice'] = product_data.select(F.variance(F.col(medianImputedPrice_column))).head()[0]
    res['numNulls_medianImputedPrice'] = product_data.filter(F.col(medianImputedPrice_column).isNull()).count()
    res['numUnknowns_unknownImputedTitle'] = product_data.filter(F.col(unknownImputedTitle_column)== 'unknown').count()





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
    product_processed_data_output = product_processed_data.withColumn(titleArray_column, F.split(F.lower(F.col(title_column)), " "))
    model = M.feature.Word2Vec(numPartitions = 4, minCount = 100, seed=102, vectorSize = 16, inputCol=titleArray_column)
    model = model.fit(product_processed_data_output)
    




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
    res['count_total'] = product_processed_data_output.count()
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
    indexer = M.feature.StringIndexer(inputCol="category", outputCol=categoryIndex_column).fit(product_processed_data)
    y = indexer.transform(product_processed_data)
    encoder = M.feature.OneHotEncoderEstimator(inputCols = [categoryIndex_column], outputCols = [categoryOneHot_column], dropLast = False)
    model = encoder.fit(y)
    y = model.transform(y)
    pca = M.feature.PCA(k=15, inputCol = categoryOneHot_column, outputCol = categoryPCA_column)
    model = pca.fit(y)
    y = model.transform(y)
    count_total = y.count()
    meanVector_categoryOneHot = y.select(M.stat.Summarizer.mean(F.col(categoryOneHot_column))).head()[0]
    meanVector_categoryPCA = y.select(M.stat.Summarizer.mean(F.col(categoryPCA_column))).head()[0]





    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'meanVector_categoryOneHot': [None, ],
        'meanVector_categoryPCA': [None, ]
    }
    # Modify res:

    res = {
        'count_total': count_total,
        'meanVector_categoryOneHot': meanVector_categoryOneHot,
        'meanVector_categoryPCA': meanVector_categoryPCA
    }


    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_6')
    return res
    # -------------------------------------------------------------------------
    
    
def task_7(data_io, train_data, test_data):
    
    # ---------------------- Your implementation begins------------------------
    
    dt = M.regression.DecisionTreeRegressor(featuresCol="features", maxDepth=5, labelCol='overall')
    model = dt.fit(train_data)
    result = model.transform(test_data)
    evaluator = M.evaluation.RegressionEvaluator(predictionCol='prediction', labelCol = 'overall')
    rmse = evaluator.evaluate(result)   
    
    
    
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
    rmses = []
    for n in [5,7,9,12]:
        (trainingData, validationData) = train_data.randomSplit([0.75, 0.25])
        dt = M.regression.DecisionTreeRegressor(featuresCol="features", maxDepth=n, labelCol='overall')
        model = dt.fit(trainingData)
        result = model.transform(validationData)


        evaluator = M.evaluation.RegressionEvaluator(predictionCol='prediction', labelCol = 'overall')
        rmses.append(evaluator.evaluate(result))
   
    
    
    
    
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
    res['test_rmse'] = rmses[0]
    res['valid_rmse_depth_5'] = rmses[0]
    res['valid_rmse_depth_7'] = rmses[1]
    res['valid_rmse_depth_9'] = rmses[2]
    res['valid_rmse_depth_12'] = rmses[3]

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_8')
    return res
    # -------------------------------------------------------------------------

