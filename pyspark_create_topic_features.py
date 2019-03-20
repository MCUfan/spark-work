####################### Import Packages ########################
import pandas as pd
import numpy as np
import re
import string
import csv
#import os
#import collections
#import pyspark
#from fuzzywuzzy import fuzz
from collections import Counter
from difflib import SequenceMatcher

from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from pyspark.sql import SparkSession, SQLContext
from pyspark import SparkContext
from pyspark.sql import Row
from pyspark.sql.functions import udf, lit, broadcast, explode, col
from pyspark.sql.types import StringType, IntegerType, LongType, MapType, ArrayType


from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

####################### Global Variables ######################
INPUT_APP_DESC = "cw.app_desc_text_data"
INPUT_APP_SEARCHES = "cw.app_searches_data"

OUTPUT_APP_TAGS = "cw.app_desc_topic_features"
OUTPUT_APP_SEARCHES = 'cw.app_desc_topic_features'

spark = SparkSession\
    .builder\
    .appName("app_rec")\
    .enableHiveSupport()\
    .getOrCreate()

NUM_TOPICS = 100
NUM_TOP_WORDS = 50



############################# Import Data ##############################
app_tag_data1 = spark.sql("select * from " + INPUT_APP_TAGS )
app_search_data = spark.sql("select * from " + INPUT_APP_SEARCHES)

app_tag_data1 = app_tag_data1.select(app_tag_data1.masterappid.cast("string")
    , app_tag_data1.apptoken
    , app_tag_data1.companyname
    , app_tag_data1.app_name
    , app_tag_data1.metakeywords
    , app_tag_data1.metadescription
    , app_tag_data1.tagline)

app_tag_data = app_tag_data1.toPandas()
app_search_data = app_search_data1.toPandas()
app_tag_data = app_tag_data['masterappid'].astype('str')


############################## Vectorize App Tag Data ############################
stop_words = stopwords.words('english')
stemmer = SnowballStemmer("english").stem

bc_stopwords = sc.broadcast(stop_words)
bc_stemmer = sc.broadcast(stemmer)

def rmPuncNum(stringIn):
    # Deal with the NoneType situation
    if stringIn is None:
        stringIn = ''
    # Create a list include all punctuation
    punc_num = string.punctuation
    regex = re.compile("[%s]" % re.escape(punc_num))
    stringOut = regex.sub(' ', stringIn)
    # Remove stand alone numbers
    stringOut = re.sub(r'\b\d+(?:\.\d+)?\s+', " ", stringOut)
    # Strip extra spaces
    stringOut = re.sub(r"\s+", " ", stringOut)
    # Trim leading/trailing spaces
    stringOut = stringOut.strip()
    return stringOut


def smart_tokenize(string):
    if isinstance(string, str):
    # convert all letters to lowercase
        string = str.lower(string)
        # remove punctuations and stand-alone numbers
        string = rmPuncNum(string)
        #string = string.strip(punctuation)
        
        # tokenize string
        tokens = re.findall(r"[\w'-]+", string)
        
        # remove stopwords
        tokens = [w for w in tokens if not w in bc_stopwords.value]
        
        # word stemming
        tokens = list(map(bc_stemmer.value, tokens))
    else:
        tokens = []
    return tokens

smart_tokenize_F = udf(lambda x: smart_tokenize(x), ArrayType(StringType()))

def stringTokenizer(tokens):
    stop_words = stopwords.words('english')
    stemmer = SnowballStemmer("english").stem
    if pd.isnull(tokens):
        tokens = ""
    return smart_tokenize(tokens, stop_words, stemmer)

def combine_text(strings):
    strings = map(str, strings)
    return ' '.join(strings)

# combine all text fields and tokenize them
app_tag_data = app_tag_data.fillna('  ')
app_tag_data['all_text'] = app_tag_data[['companyname', 'application_name','metakeywords','metadescription', 'tagline']].apply(combine_text, axis = 1)
app_tag_data['tokens'] = app_tag_data['all_text'].apply(smart_tokenize)


# convert combined text field to TF-IDF
tfidfVectorizer = TfidfVectorizer(norm=None, \
                                  use_idf=True, \
                                  smooth_idf=True, \
                                  sublinear_tf=False, \
                                  analyzer = "word",   \
                                  tokenizer = smart_tokenize,    \
                                  preprocessor = None, \
                                  max_features = 50000)

tfidf = tfidfVectorizer.fit_transform(app_tag_data['all_text'])
tfidf_feature_names = tfidfVectorizer.get_feature_names()

# topic modeling using TF_IDF

# Run NMF
nmf = NMF(n_components = NUM_TOPICS, random_state =1, alpha = 0.1, l1_ratio = 0.5, init = 'nndsvd').fit(tfidf)

#Run LDA
lda = LatentDirichletAllocation(n_topics = NUM_TOPICS, max_iter = 5, learning_method = 'online', learning_offset = 50, 
                               random_state = 0).fit(tfidf)

## output app_tag_features
nmf_features = pd.DataFrame( nmf.transform(tfidf) )
nmf_features.columns = [('topic_' + str(column)) for column in nmf_features.columns.values]
COLUMNS_TO_DROP = ['companyname', 'application_name', 'metakeywords', 'metadescription', 'tagline'
                   ,'all_text', 'tokens']
app_tag_features = pd.concat([app_tag_data, nmf_features], axis=1).drop(COLUMNS_TO_DROP, axis = 1)
app_tag_features = spark.createDataFrame(app_tag_features)

app_tagg_vector_features = app_tag_features.createOrReplaceTempView("app_tagg_vector_features")
spark.sql("drop table if exists " + OUTPUT_APP_TAGS)
spark.sql("create table " + OUTPUT_APP_TAGS  + " stored as parquet as select * from app_tagg_vector_features")


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def topic_score(value, token):
    scores = list(map(similar, value.split(" "), [token] * len(value) ))
    return int(max(scores) > 0.8)

empty_dic = {}
for i in range(NUM_TOPICS):
   empty_dic[i] = 0

bc_empty_dic = sc.broadcast(empty_dic)

def vectorize_searches(searches, topics):
    if searches == []:
        return bc_empty_dic.value
    else:
        search_dic = Counter({})
        for token in searches: 
            token_dic = Counter({})
            for key, value in topics.items():
                token_dic[key] = topic_score(value, token)
            search_dic = search_dic + token_dic
        feature_dic = {}
        for topic in range(no_topics):
            feature_dic[topic] = 0 + search_dic[topic]
        return feature_dic

def vectorize_searches_F(topics):
    return udf(lambda x: vectorize_searches(x, topics), MapType(IntegerType(), IntegerType()))

def display_topics(model, feature_names, no_top_words):
    output_dic = {}
    for topic_idx, topic in enumerate(model.components_):
        output_dic[topic_idx] = " ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])
    return output_dic


nmf_topics = display_topics(nmf, tfidf_feature_names, NUM_TOP_WORDS)
bc_nmf_topics = sc.broadcast(nmf_topics)


app_search_data1 = app_search_data
app_search_data1 = app_search_data1.withColumn('search_tokens', smart_tokenize_F(app_search_data1.search_keywords_collection)).select('company_id', 'upper_bound_dt', 'date_for_day', 'avg_nbr_searches', 'search_tokens')
app_search_data1 = app_search_data1.withColumn("date_for_day", app_search_data1["date_for_day"].cast(StringType()))
app_search_data1 = app_search_data1.withColumn("search_vectors", vectorize_searches_F(bc_nmf_topics.value)(app_search_data1.search_tokens))

app_search_data2 = app_search_data1.toPandas()
app_search_data2 = pd.concat([app_search_data2[['company_id', 'upper_bound_dt','date_for_day','avg_nbr_searches']] ,pd.DataFrame(list(app_search_data2['search_vectors']))] , axis = 1)

column_names = ['company_id', 'upper_bound_dt','date_for_day','avg_nbr_searches']
column_names = column_names + ['topic_'+ str(i) for i in range(no_topics)]
app_search_data2.columns = column_names 

search_vectorized = spark.createDataFrame(app_search_data2)

app_search_vector_features = search_vectorized.createOrReplaceTempView("app_search_vector_features")
spark.sql("drop table if exists " + OUTPUT_APP_SEARCHES)
spark.sql("create table " + OUTPUT_APP_SEARCHES + " stored as parquet as select * from app_search_vector_features")