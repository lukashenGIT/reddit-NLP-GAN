# --- Dependencies ---
import numpy as np
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.functions import udf
from pyspark.sql.types import *

# --- User defined dependencies ---
from parseHelper import *


#To print entire np matrices, decomment:
#np.set_printoptions(threshold=np.nan)


def main():
    fileName = "./resources/dataset_chunk.json"
    conf = SparkConf().setMaster("local").setAppName("My_App")
    sc = SparkContext(conf = conf)
    spark = SQLContext(sc)
    json_df = spark.read.json(fileName)


    rdd = (
        json_df.select("body", "ups")
        .rdd
        .filter(lambda l: "[deleted]" not in l["body"])
        .filter(lambda l: "[removed]" not in l["body"])
        .filter(lambda l: l["ups"] > 10)
        #.map(lambda l: print(l["body"]))
        )

    #Define get functions from parseHelper as pyspark UDF (requires pyspark.sql.functions.udf).
    udfToOneHot = udf(toOneHot)
    udfToString = udf(toString)

    df = spark.createDataFrame(rdd)
    df_onehot = df.withColumn("body_onehot", udfToOneHot("body"))
    df_onehot.show()



if __name__ == '__main__':
    main()
