#### Step 2b: clean email, phone numbers and addresses 

import pandas as pd
import numpy as np
import re
import os
import collections
#import pyspark

from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql import Row
from pyspark.sql.functions import udf, lit, monotonically_increasing_id, broadcast
from pyspark.sql.types import StringType, IntegerType, LongType

from validate_email import validate_email
import twilo
import phonenumbers

####################### Global Variables ######################
INPUT_TABLE_NAME = "cw.contacts_data"
OUTPUT_TABLENAME = "cw.contacts_data_cleaned"
COUNT_LIMIT = 50

####################### Create Spark Session ##################
spark = SparkSession\
	.builder\
	.appName("QBO Map")\
	.enableHiveSupport()\
	.getOrCreate()

############################## UDF'S ###############################           
# function to convert emails from badEmailSet to nulls, take only first email 
# and limit it to 50 char's or less

def cleanEmail(x, badEmailSet):
	if valid_email(x) is False:
		return None
    if (x in badEmailSet) or (x is None):
        return None
    else:
        first_email = x.split(',')[0]
        first_email = "".join(first_email.split())
        if len(first_email) > 50:
            return None
        else:
            return first_email


# wrapper function for cleanEmail to be used with pyspark dataframe
def cleanEmail_F(badEmailSet):
	return udf(lambda x: cleanEmail(x, badEmailSet), StringType())

# extract phone number including extension 
def extract_phone_number(x):
    phonePattern = re.compile(r'''
                    # don't match beginning of string, number can start anywhere
        (\d{3})     # area code is 3 digits 
        \D*         # optional separator is any number of non-digits
        (\d{3})     # trunk is 3 digits 
        \D*         # optional separator
        (\d{4})     # rest of number is 4 digits 
        \D*         # optional separator
        (\d*)       # extension is optional and can be any number of digits
        $           # end of string
        ''', re.VERBOSE)
    return phonePattern.search(x)

# function to clean phone numbers
def cleanPhoneNum(x, badPhoneSet):
# filter out bad phone numbers
	if x in badPhoneSet:
		return None
    elif extract_phone_number(x) is None:
    	return None
	else:
        numbers = ''.join(extract_phone_number(x).groups())
		# filter out numbers with same digits appearing more than 6 times
		if collections.Counter(numbers).most_common(1)[0][1] > 6:
			return None
		else: 
			return numbers

#wrapper function for cleanPhoneNum to be used with pyspark dataframe      
def cleanPhoneNum_F(badPhoneSet):
	return udf(lambda x: cleanPhoneNum(x, badPhoneSet), StringType())

# standardize address using google map api
def google_address(address):
    try:
        g_address = gmaps.geocode(address)
    except:
        print(address + ' is not a proper address')
    if g_address == []:
        address_c = address
    elif len(g_address) > 1:
        address_c = address
    else:
        address_c = g_address[0]['formatted_address']
    return address_c



############################# Import Data ##############################
#df = spark.sql("select email1, email2, phone1, phone2 from " + INPUT_TABLE_NAME)
df = spark.sql("select company_id, entity_id, entity_name, name1, name2, company_name1, first_name1, last_name1, address1, city1, state1, postal_code1 ,email1, email2, phone1, phone2 from " + INPUT_TABLE_NAME)


########################## get sets of over-frequent e-mails, phones, addresses ##########
# create dataframe for most frequent email1's and their frequency
email1_freq = df.select('email1').groupby('email1').count().orderBy('count', ascending = False).withColumnRenamed("count", "freq")
# create set of email1's with frequency > 50
bad_email1 = set(email1_freq.filter((email1_freq.freq > COUNT_LIMIT) & (email1_freq.email1.isNotNull())).select('email1').rdd.flatMap(lambda x: x).collect())


# create dataframe for most frequent email2's and their frequency
email2_freq = df.select('email2').groupby('email2').count().orderBy('count', ascending = False).withColumnRenamed('count', 'freq')
# create set of email2's with frequency > 50
bad_email2 = set(email2_freq.filter((email2_freq.freq > COUNT_LIMIT) & (email2_freq.email2.isNotNull())).select('email2').rdd.flatMap(lambda x: x).collect())


# create dataframe for most frequent phone1's and their frequency
phone1_freq = df.select('phone1').groupby('phone1').count().orderBy('count', ascending = False).withColumnRenamed('count', 'freq')
# create set of phone1's with frequency > 50
bad_phone1 = set(phone1_freq.filter((phone1_freq.freq > COUNT_LIMIT) & (phone1_freq.phone1.isNotNull())).select('phone1').rdd.flatMap(lambda x: x).collect())

# create dataframe for most frequent phone2's and their frequency
phone2_freq = df.select('phone2').groupby('phone2').count().orderBy('count', ascending = False).withColumnRenamed('count', 'freq')
#create set of phone2's with frequency > 50 
bad_phone2 = set(phone2_freq.filter((phone2_freq.freq > COUNT_LIMIT) & (phone2_freq.phone2.isNotNull())).select('phone2').rdd.flatMap(lambda x: x).collect())

########################### Clean Data #####################
#clean email1, email2, phone1, phone2, address
df2 = df.withColumn("email1_c", cleanEmail_F(bad_email1)(df.email1))
df2 = df2.withColumn('email2_c', cleanEmail_F(bad_email2)(df.email2))

df2 = df2.withColumn('phone1_c', cleanPhoneNum_F(bad_phone1)(df.phone1))
df2 = df2.withColumn('phone2_c', cleanPhoneNum_F(bad_phone2)(df.phone2))
#df2 = df2.withColumn('address1_c', cleanAddress_F(bad_address1)(df.address1))

# convert phone number to integer
df2 = df2.withColumn('phone1_c', df2.phone1_c.cast(LongType()))
df2 = df2.withColumn('phone2_c', df2.phone2_c.cast(LongType()))

#filter out rows with too many nulls
df2 = df2.withColumn('role_id', monotonically_increasing_id())
df2 = df2.filter(df2.email1_c.isNotNull() | df2.email2_c.isNotNull() | df2.phone1_c.isNotNull() | df2.phone2_c.isNotNull())


######################### Write Cleaned Data Back to Hive ####################
contacts_data_cleaned = df2.createOrReplaceTempView("contacts_data_cleaned")

spark.sql("drop table if exists " + OUTPUT_TABLENAME)
spark.sql("create table " + OUTPUT_TABLENAME + " stored as parquet as select * from contacts_data_cleaned")