#!/usr/bin/env python

# Importing all required libraries
"""
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

"""

eda_op_dict={}# dictionary is global 
def date_time(datetime_str):
    return datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
def outlier_treatment_process(df,c):
    q1,q3 = df.approxQuantile(c, [0.25, 0.75], 0)
    iqr = q3 - q1
    minimum = q1 - (iqr * 1.5)
    maximum = q3 + (iqr * 1.5)
    return minimum

def discrete_count_plot(result,column):
    mpl.style.use('seaborn')
    plt.bar(range(len(result)), list(result.loc[:,'count'].values), align='center')
    plt.xlabel(column,fontsize=15)
    plt.ylabel('Counts',fontsize=15)
    plt.xticks(range(len(result)), list(result.loc[:,column].values),fontsize=15,rotation=65, horizontalalignment='right')
    plt.title('Histogram of '+column,fontsize=15)
    plt.savefig(column+'.png',bbox_inches = "tight")
     

def continuous_count_plot(result,column,labels=None):
    mpl.style.use('seaborn')
    if labels is None:
        labels=list(result.loc[:,'labels'].values)
        
    #plt.figure(figsize=(16,300))
    plt.bar(range(len(result)), list(result.loc[:,'count'].values), align='center')
    plt.xlabel(column,fontsize=15)
    plt.ylabel('Counts',fontsize=15)
    plt.xticks(range(len(result)), labels,fontsize=15,rotation=65, horizontalalignment='right')
    plt.title('Histogram of '+column,fontsize=15)
    plt.savefig(column+'.png',bbox_inches = "tight")


def get_continuous_lab(unlabel_list,inputCol,outputCol,query_frame,get_scientfic=False):
    label_df=pd.DataFrame()
    label_list=[]
    for i in unlabel_list:
        re=query_frame.filter(col(outputCol).isin([i]))
        mini=re.agg({inputCol: "min"}).collect()[0]['min('+inputCol+')']
        if get_scientfic:
            if mini is not None:
                mini='%e' % mini
        maxm=re.agg({inputCol: "max"}).collect()[0]['max('+inputCol+')']
        if get_scientfic:
            if maxm is not None:
                maxm='%e' % maxm
        label_list.append(str(mini)+'-'+str(maxm))
    label_df[outputCol]=unlabel_list
    label_df['labels']=label_list
    return label_df

def discrete_univariate_analysis(data,column,max_cat_len=5,save_csv=False):
        df=data.groupBy(column).count().toPandas()
        #sorting in descending
        df=df.sort_values(by='count',ascending=False)
        if len(df)>max_cat_len:
            #taking required max length 
            df_max_cat_len=df[:max_cat_len]
            #making dict out of df
            dict_df_max=dict(zip(df.loc[:,column], df.loc[:,'count']))
            #converting the keys to str
            dict_df_max_string = dict([(str(k), v) for k, v in dict_df_max.items()])
            #saving values of nan 
            if ('nan' in dict_df_max_string):
                df_nan=pd.DataFrame({column:'nan','count':dict_df_max_string['nan']},index=[1])
            else:
                df_nan=pd.DataFrame()
            #taking sum of other values and putting in other category
            sum_other_category_val=pd.DataFrame({column:'others','count':sum(df.loc[max_cat_len:,'count'])},index=[1])
            #concatenating all of them
            df_shorted_max = pd.concat([df_max_cat_len,df_nan,sum_other_category_val])
            #plotting dataframe
            discrete_count_plot(df_shorted_max,column)
            if save_csv:
                df_shorted_max.to_csv(column+'.csv',index=False)
            # Update the dictionary
            #eda_op_dict.update({'chart_id'+str(len(eda_op_dict)):{'column_name':column,'img_path':os.getcwd()+'/'+column+'.png','count':df}})
        else:
            #plotting dataframe
            discrete_count_plot(df,column)
        # Update the dictionary
            if save_csv:
                df.to_csv(column+'.csv',index=False)
        eda_op_dict.update({'chart_id'+str(len(eda_op_dict)):{'column_name':column,'img_path':os.getcwd()+'/'+column+'.png','count':df}})

def continuous_univariate_analysis(data,column,Buckets=5,outlier_treatment=False,save_csv=False,get_scientfic_val=False):
    out_col="bucket_labels"# just given a hardcoded naming to output of label column,this does not affect code 
    # here I am selecting the column 
    data_col=data.select(column)
    # Treating the outlier in left of ditribution anything less than q1 - (iqr * 1.5) is removed  .
    if outlier_treatment:
        minimum=outlier_treatment_process(data,column)
        data_col=data_col.filter(column+'>'+str(minimum))
        
     
    #QuantileDiscretizer is used when no labels are passed and only number of buckets needed are given 
    # or by default 5 is what I have taken for this function , here separate column for nan will be formed 
    # as I have used handleInvalid parameter, this gives us labels for each values in column
    discretizer = QuantileDiscretizer(numBuckets=Buckets, inputCol=column, outputCol=out_col,handleInvalid="error")
    result_buckets = discretizer.setHandleInvalid("keep").fit(data_col).transform(data_col)
    # we get the labels from QuantileDiscretizer and grouby by the labels to get count for each labels
    label_count_df=result_buckets.groupBy(out_col).count().toPandas().sort_values(by=out_col,ascending=True)
    # now we need to know which labels represents which range so we find out firstly distinct labels
    labels_distinct=list(label_count_df[out_col])
    # now get_continuous_lab helps us to get the min and max values of particular label and gives list of 
    #labels as output
    output_labels=get_continuous_lab(labels_distinct,column,out_col,result_buckets,get_scientfic=get_scientfic_val)
    merged_df_labels=pd.merge(output_labels,label_count_df,on=out_col,how="inner")
    #print(merged_df_labels)
    continuous_count_plot(merged_df_labels,column)# plotting the dataframe
    # Update the dictionary
    if save_csv:
        merged_df_labels.to_csv(column+'.csv',index=False)
    
    eda_op_dict.update({'chart_id'+str(len(eda_op_dict))+'_continuous_data':\
    {'column_name':column,'img_path':os.getcwd()+'/'+column+'.png'},'count_frame':label_count_df})

def get_trend_df(data,date_col_name,tran_amountcol_name,agg_type="sum",frequency_window='1 day',key2_groupby=False,key2=None,frequency_period='D',to_csvopt=False):
    ## Here depending on if user wants groupby with some specific key or specific aggregation 4 cases are written
    if agg_type=="avg"and key2_groupby==True:
        wingrp_result=data.groupBy(key2,window(date_col_name, frequency_window)).agg(avg(tran_amountcol_name).alias("avg"))
        result_win=wingrp_result.select(key2,wingrp_result.window.start.cast("string").alias("start"),wingrp_result.window.end.cast("string").alias("end"), agg_type).toPandas()
    if agg_type=="sum"and key2_groupby==True:
        wingrp_result=data.groupBy(key2,window(date_col_name, frequency_window)).agg(_sum(tran_amountcol_name).alias("sum"))
        result_win=wingrp_result.select(key2,wingrp_result.window.start.cast("string").alias("start"),wingrp_result.window.end.cast("string").alias("end"), agg_type).toPandas()
    if agg_type=="sum" and key2_groupby==False:
        wingrp_result=data.groupBy(window(date_col_name, frequency_window)).agg(_sum(tran_amountcol_name).alias("sum"))
        result_win=wingrp_result.select(wingrp_result.window.start.cast("string").alias("start"),wingrp_result.window.end.cast("string").alias("end"), agg_type).toPandas()
    if agg_type=="avg"and key2_groupby==False:
        wingrp_result=data.groupBy(window(date_col_name,frequency_window)).agg(avg(tran_amountcol_name).alias("avg"))
        result_win=wingrp_result.select(wingrp_result.window.start.cast("string").alias("start"),wingrp_result.window.end.cast("string").alias("end"), agg_type).toPandas()
    ## Applying date_time function for format conversion of date columns 
    result_win['end_date']=result_win['end'].apply(date_time)
    result_win['start_date']=result_win['start'].apply(date_time)
    sort_result_win=result_win.sort_values(by='start_date',ascending=True).reset_index()
    ## Making dataframe consisting all dates from start to end of dates present in sort_result_win dataframe to fill in missing dates
    time=pd.DataFrame(pd.date_range(start=sort_result_win.start_date[0], end=sort_result_win.start_date[sort_result_win.shape[0]-1],freq=frequency_period),columns=['start_date'])
    time['end_date']=pd.DataFrame(pd.date_range(start=sort_result_win.end_date[0], end=sort_result_win.end_date[sort_result_win.shape[0]-1],freq=frequency_period))
    # Merging the sort_result_win and time df to get final output
    merge_results=pd.merge(time,sort_result_win,how='outer',on=['start_date','end_date'])
    merge_results.drop(['end','start','index'],inplace=True,axis=1)
    merge_results.fillna(0,inplace=True)
    if to_csvopt:
        merge_results.to_csv('result_trent_plot.csv',index=False)
    return merge_results

