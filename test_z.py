# Databricks notebook source
pip install xlrd

# COMMAND ----------

import numpy as np
import pandas as pd
import xlrd
from IPython.display import display
pd.options.display.max_columns = None
from pyspark.sql.functions import *
import pyspark.sql.functions as sf
from pyspark.sql import functions as F
from pyspark.sql.functions import substring, length, col, expr, mean, split, avg, lower
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, LongType, StringType, FloatType
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext

# COMMAND ----------

from functools import reduce
from pyspark.sql import DataFrame

# COMMAND ----------

def unionAll(*dfs):
    return reduce(DataFrame.unionAll, dfs)

# COMMAND ----------

test1 = spark.sql(""" select * from sales_csv """)

# COMMAND ----------

test1.write.format("com.databricks.spark.csv").option("header", "true").save("/FileStore/tables/test1.csv")

# COMMAND ----------

dbfs cp "dbfs:/FileStore/tables/sales.csv" "C:\Users\opti1431\Downloads\sales.csv"

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

vectorAssembler = VectorAssembler(inputCols = ['rate', 'sales_in_first_month', 'sales_in_second_month'], outputCol = 'features')

# COMMAND ----------

spark.sql("show tables").show()

# COMMAND ----------

test1.show()

# COMMAND ----------

test1 = test1.withColumn("rate", test1["rate"].cast(IntegerType()))
test1 = test1.withColumn("sales_in_first_month", test1["sales_in_first_month"].cast(IntegerType()))
test1 = test1.withColumn("sales_in_second_month", test1["sales_in_second_month"].cast(IntegerType()))
test1 = test1.withColumn("sales_in_third_month", test1["sales_in_third_month"].cast(IntegerType()))

# COMMAND ----------

dataset_v1 = vectorAssembler.transform(test1)

# COMMAND ----------

dataset_v2 = dataset_v1.select(['features', 'sales_in_third_month'])

# COMMAND ----------

dataset_v2.show()

# COMMAND ----------

lr = LinearRegression(featuresCol = 'features', labelCol = 'sales_in_third_month')

# COMMAND ----------

lr_model = lr.fit(dataset_v2)

# COMMAND ----------

print("Coefficient: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

# COMMAND ----------

str(lr_model.coefficients)

# COMMAND ----------

lst = [str(lr_model.coefficients), str(lr_model.intercept)]

# COMMAND ----------

lst

# COMMAND ----------

test_results = pd.DataFrame(lst)

# COMMAND ----------

mySchema = StructType([StructField("results", StringType(), True)])

# COMMAND ----------

test_results_spark = spark.createDataFrame(test_results, schema = mySchema)

# COMMAND ----------

test_results_spark.show()

# COMMAND ----------

pip install spark-sklearn

# COMMAND ----------

from spark_sklearn import Converter

# COMMAND ----------

converter = Converter(spark.sparkContext)
sk_model = converter.toSKLearn(lr_model)

# COMMAND ----------

import pickle

# COMMAND ----------

pickle.dump(sk_model, open('model2.pkl','wb'))

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------
