# --- Dependencies ---
import numpy as np
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.functions import udf, col, size
from pyspark.sql.types import *
from pyspark.ml.feature import RegexTokenizer
from pyspark.sql.types import *
from pyspark.sql import *
import tensorflow as tf
import sys
import ast

# --- User defined dependencies ---
from parseHelper import *


#To print entire np matrices, decomment:
#np.set_printoptions(threshold=np.nan)


def pad(vec):
    """ padding of vector. pad length defined here """
    pad_len = 10 # number of words to pad to.
    if len(vec) < pad_len:
        vec.extend([0]*(pad_len - len(vec)))
    elif len(vec) > pad_len:
        vec = vec[:pad_len]
    return vec

def _int64_feature(value):
    """ for writing int to tfrecord """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def main():

    """ Configure Spark """
    fileName = "./resources/dataset_chunk.json"
    conf = SparkConf().setMaster("local[*]").setAppName("My_App")
    sc = SparkContext(conf = conf)
    sqlC = SQLContext(sc)

    """ Read Dataframe """
    json_df = sqlC.read.json(fileName)

    """ Filter out irrelevant data """
    df = json_df.select("body", "ups")\
        .filter("[deleted]" != json_df["body"])\
        .filter("[removed]" != json_df["body"])\
        .filter(json_df["ups"] > 5)

    """ Tokenize comments with regexp. matches anything but chars/nums and ' """
    rtokenizer = RegexTokenizer(inputCol="body", outputCol="tokenized", \
    pattern="\\s+|[^(a-z)|^(A-Z)|^(0-9)|^(\\’\\'\\´)]|[()]")

    df_tokenized = rtokenizer.transform(df)


    """ Make RDD and flatten to WordList (filter out comments with more than PAD_LEN words) """
    df_tok_filt = df_tokenized.filter(size("tokenized") <= 10)
    tokensRDD = df_tok_filt.select("tokenized").rdd
    wordListRDD = tokensRDD.flatMap(list).flatMap(lambda x: x).distinct()

    """ Show wordlist & Number of entries """
    #print(wordListRDD.collect())
    #print(wordListRDD.count())

    """ zip rdd to tuples of (word,index) """
    zippedRDD = wordListRDD.zipWithIndex()

    """ collect as local dictionary and broadcast"""
    wordList = zippedRDD.collectAsMap()

    """ Increase index by 1 so that pad can be 0 and add pad"""
    wordList.update((k, v + 1) for k,v in wordList.items())
    wordList["§PAD$"] = 0

    """ Make UDF that makes vecs from wordList and add to previous df. """
    wordList_bc = sc.broadcast(wordList)
    sent2vecUDF = udf(lambda x: [wordList_bc.value[word] for word in x])
    df_vec = df_tok_filt.withColumn("vectors", sent2vecUDF("tokenized"))

    """ make pad udf and pad/cut vectors """
    padUDF = udf(pad)
    df_padded = df_vec.withColumn("vec_padded", padUDF("vectors"))
    #df_padded.show()

    """ view lengths of padded comments (check padding) """
    #testUDF = udf(lambda x: len(x))
    #df_test = df_padded.withColumn("counts", testUDF("vec_padded"))
    #df_test.show()

    df_final = df_padded.drop("body", "vectors", "tokenized")
    #df_final.show()

    """ Make iterator for vectors and upvotes (features)"""
    vec_it = df_final.select("vec_padded").rdd.toLocalIterator()
    ups_it = df_final.select("ups").rdd.toLocalIterator()

    """ use local iterators to write to tfrecords """
    train_filename = "train.tfrecords"
    test_filename = "test.tfrecords"

    """ open tfrecord file """
    writer = tf.python_io.TFRecordWriter(train_filename)

    """ iterate over rdd's and write to tfrecord """

    for vec, ups in zip(vec_it, ups_it):

        sentence = ast.literal_eval(vec["vec_padded"]) #literal eval since padded vecs are casted to strings somewhere. (unintentional)
        #label = [ups["ups"]]

        example = tf.train.Example(features=tf.train.Features(feature={
        'sentence': _int64_feature(sentence)}))

        #example = tf.train.Example(features=tf.train.Features(feature={
        #'sentence': _int64_feature(sentence),
        #'label': _int64_feature(label)}))

        writer.write(example.SerializeToString())

    writer.close()






if __name__ == '__main__':
    main()
