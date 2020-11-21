## EDA Function Documentation

There are three main functions and five sub functions used by these main functions which I am describing below:

Always make a eda_op_dict={} before using this function this saves few important parameter in dict which can be later used in session

1) discrete_univariate_analysis : for discrete data analysis
	discrete_univariate_analysis(data,column,max_cat_len=5,save_csv=False)
	
data: dataframe to be passed ,read the csv in save in variable and pass it in function

column: name the column to be analysed now I have made it to analyse only one at a time if there a requirement we can pass list of 
column names put a for loop inside the function or call the function in loop 

max_cat_len: it comes into picture if your discrete column has more number of variables may be some are not so important and less occuring this will bucket them and add into others column and plot them but in memory original result is saved to refer.By default 5 in max number of classes
to be considered .

save_csv: If true will save the result dataframe .

## This function is dependent on discrete_count_plot for plotting and saving  the graph in .png format.

2) continuous_univariate_analysis: for continuous data analysis

continuous_univariate_analysis(data,column,Buckets=5,outlier_treatment=False,save_csv=False,get_scientfic_val=False) 

data: dataframe to be passed ,read the csv in save in variable and pass it in function

column: name the column to be analysed now I have made it to analyse only one at a time if there a requirement we can pass list of 
column names put a for loop inside the function or call the function in loop 

outlier_treatment: it treats outlier by clipping the anything less than q1 - (iqr * 1.5) is removed it uses the outlier_treatment_process function.

Buckets: number of buckets you want your function to be divided

save_csv: If true will save the result dataframe 

get_scientfic_val : If you want scientific names for your plot labels

## This function has dependency on continuous_count_plot for plotting and saving  the graph in .png format,outlier_treatment_process for outlier treatment and get_continuous_lab for generating labels for continuos plot .


3) get_trend_df :for getting trend dataframe according to time frame passed 

get_trend_df(data,date_col_name,tran_amountcol_name,agg_type="sum",frequency_window='1 day',key2_groupby=False,key2=None,frequency_period='D',to_csvopt=False)\

data: dataframe to be passed ,read the csv in save in variable and pass it in function

date_col_name:Column name containing date using which you want to slice the data

tran_amountcol_name: Column name containing amount to be used to see trend

agg_type: Aggreagation wanted , currently supports only average and sum , more can be added

frequency_window: slicer for the data . Durations are provided as strings, e.g. ‘1 second’, ‘1 day 12 hours’, ‘2 minutes’. Valid interval strings are ‘week’, ‘day’, ‘hour’, ‘minute’, ‘second’, ‘millisecond’, ‘microsecond’. If the slideDuration is not provided, the windows will be tumbling windows.

Please refer https://spark.apache.org/docs/latest/api/python/pyspark.sql.html?highlight=window#pyspark.sql.functions.window

key2_groupby: If need another key for groupby along with frequency_window. If none then data will be grouped by frequency_window only .

key2: By default it is none but if you activate key2_groupby need to give key2 else will throw error

frequency_period : for time_period function to generate missing dates have to give same frequency as frequency_window 
e.g if frequency_window="2 hour" so  frequency_period="2h"

see this link for refernce https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases

to_csvopt : to save the output to csv else the function will give dataframe as output

e.g I have data on transaction per day , customer does transaction 4-5 times per day and some days none.I want to see it in slot of 2 hours 
I will use the function as :

df=get_trend_df(new_data,"parsed_date","tran_amount",agg_type="avg",frequency_window="2 hour",key2_groupby=True,key2="response",frequency_period='2h',to_csvopt=True)

##  This function has dependency on date_time function for date conversions in pandas dataframe.

See trend_function_demo_cg.ipynb for the get_trend_df function demo .
See Univariate_EDA_functions.ipynb for the demo of continuous and discrete functions

Please install required libraries before using the function , details can be found in requirements.txt

# Importing all required libraries

from pyspark.sql import SparkSession
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import os
from pyspark.ml.feature import Bucketizer
from pyspark.ml.feature import QuantileDiscretizer
import pandas as pd
import numpy as np
from math import isnan
import pyspark
import pyspark.sql.functions as f
import matplotlib as mpl
from pyspark.sql.functions import window
from pyspark.sql.functions import sum as _sum
from pyspark.sql.functions import col
# Spark datatypes to contruct schema
from pyspark.sql.types import  (StructType, 
                                StructField, 
                                DateType, 
                                BooleanType,
                                DoubleType,
                                IntegerType,
                                StringType,
                               TimestampType)
from pyspark.sql.functions import udf
from pyspark.sql.functions import avg 
from datetime import datetime

