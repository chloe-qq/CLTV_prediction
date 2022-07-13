import re
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colorbar

import pyspark.sql.functions as F

import multiprocessing as mp

from pyspark.ml.feature import Normalizer
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import IndexToString
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import Bucketizer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.ml.feature import Imputer
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import Word2Vec
from pyspark.ml.linalg import SparseVector
from pyspark.ml import Pipeline

RANDOM_SEED = 3
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


"""
generate training data one-to-many: 12-to-12

"""
train_dt_range = ['2020-02-06','2020-03-07','2020-04-06','2020-05-06','2020-06-05','2020-07-05','2020-08-04','2020-09-03','2020-10-03','2020-11-02','2020-12-02','2021-01-01']
label_dt_range = ['2021-01-01', '2021-01-31','2021-03-02','2021-04-01','2021-05-01','2021-05-31','2021-06-30','2021-07-30','2021-08-29','2021-09-28', '2021-10-28', '2021-11-27']

DATA_PATH = '/mnt/dsnpinternship/rnn-pltv/training_data'

ID_FEATURE = ['EntityID']
TIME_FEATURE = ['snapshot_time']

CATEGORY_FEATURE = ['dataset_user_attribute_cc_dc_LATEST_ATTR_country_code',
    'dataset_user_attribute_cc_dc_LATEST_ATTR_cc_signup_source_cc_joining_date_all',
    'dataset_user_attribute_cc_dc_LATEST_ATTR_cc_signup_for_cc_joining_date_all',
    'dataset_user_attribute_cc_dc_LATEST_ATTR_dc_signup_source_dc_joining_date_all',
    'dataset_user_attribute_cc_dc_LATEST_ATTR_dc_signup_for_dc_joining_date_all',
    'business_funnel']

NUMERICAL_FEATURE = ['CLTV_30d',
    'dataset_trialist_COUNT_ACTIVE',
    'signup_age_unit_30d',
    'EventType_HAU_MOBILE_TargetType_PRODUCT_NAME_Target_SKETCH_COUNT_EventTime_L30',
    'EventType_THOR_APP_DOWNLOAD_TargetType_PRODUCT_NAME_Target_Photoshop_COUNT_EventTime_L30',
    'EventType_THOR_APP_DOWNLOAD_TargetType_PRODUCT_NAME_Target_InCopy_COUNT_EventTime_L30',
    'EventType_HAU_DESKTOP_TargetType_PRODUCT_NAME_Target_LIGHTROOM_CLASSIC_COUNT_EventTime_L30',
    'EventType_HAU_DESKTOP_TargetType_PRODUCT_NAME_Target_MEDIA_ENCODER_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_Product_Target_Photoshop_COUNT_EventTime_L30',
    'EventType_HAU_MOBILE_TargetType_PRODUCT_NAME_Target_PHOTOSHOP_COUNT_EventTime_L30',
    'EventType_THOR_APP_DOWNLOAD_TargetType_PRODUCT_NAME_Target_LightroomCC_COUNT_EventTime_L30',
    'EventType_THOR_APP_DOWNLOAD_TargetType_PRODUCT_NAME_Target_LightroomClassic_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_Domain_Target_creativecloud_adobe_com_COUNT_EventTime_L30',
    'EventType_HAU_WEB_TargetType_PRODUCT_NAME_Target_CCX_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_PagenameLevel3_Target_adobe_com:products:special-offers_COUNT_EventTime_L30',
    'EventType_THOR_APP_DOWNLOAD_TargetType_PRODUCT_NAME_Target_Bridge_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_Domain_Target_acrobat_adobe_com_COUNT_EventTime_L30',
    'EventType_HAU_MOBILE_TargetType_PRODUCT_NAME_Target_READER_COUNT_EventTime_L30',
    'EventType_THOR_APP_DOWNLOAD_TargetType_PRODUCT_NAME_Target_InDesign_COUNT_EventTime_L30',
    'EventType_HAU_DESKTOP_TargetType_PRODUCT_NAME_Target_AFTER_EFFECTS_COUNT_EventTime_L30',
    'EventType_THOR_APP_DOWNLOAD_TargetType_PRODUCT_NAME_Target_CharacterAnimator_COUNT_EventTime_L30',
    'EventType_HAU_SERVICE_TargetType_PRODUCT_NAME_Target_FONTS_COUNT_EventTime_L30',
    'EventType_HAU_DESKTOP_TargetType_PRODUCT_NAME_Target_ANIMATE_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_Product_Target_Animate_COUNT_EventTime_L30',
    'EventType_HAU_DESKTOP_TargetType_PRODUCT_NAME_Target_PREMIERE_RUSH_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_Product_Target_InDesign_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_PagenameLevel3_Target_account:activated-devices_COUNT_EventTime_L30',
    'EventType_THOR_APP_DOWNLOAD_TargetType_PRODUCT_NAME_Target_Prelude_COUNT_EventTime_L30',
    'EventType_HAU_MOBILE_TargetType_PRODUCT_NAME_Target_FRESCO_COUNT_EventTime_L30',
    'EventType_THOR_APP_DOWNLOAD_TargetType_PRODUCT_NAME_Target_Dimension_COUNT_EventTime_L30',
    'EventType_HAU_MOBILE_TargetType_PRODUCT_NAME_Target_PHOTOSHOP_FIX_COUNT_EventTime_L30',
    'EventType_THOR_APP_DOWNLOAD_TargetType_PRODUCT_NAME_Target_MediaEncoder_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_Product_Target_LightroomCC_COUNT_EventTime_L30',
    'EventType_HAU_SERVICE_TargetType_PRODUCT_NAME_Target_CLOUD-DOCS_COUNT_EventTime_L30',
    'EventType_HAU_SERVICE_TargetType_PRODUCT_NAME_Target_BEHANCE_COUNT_EventTime_L30',
    'EventType_HAU_SERVICE_TargetType_PRODUCT_NAME_Target_CCLIBRARY_COUNT_EventTime_L30',
    'EventType_HAU_DESKTOP_TargetType_PRODUCT_NAME_Target_LIGHTROOM_DESKTOP_COUNT_EventTime_L30',
    'EventType_HAU_DESKTOP_TargetType_PRODUCT_NAME_Target_PREMIERE_PRO_COUNT_EventTime_L30',
    'EventType_HAU_DESKTOP_TargetType_PRODUCT_NAME_Target_XD_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_Product_Target_Illustrator_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_PagenameLevel3_Target_ims:welcomeback:adobeid_COUNT_EventTime_L30',
    'EventType_HAU_MOBILE_TargetType_PRODUCT_NAME_Target_LIGHTROOM_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_Product_Target_PremierePro_COUNT_EventTime_L30',
    'EventType_HAU_DESKTOP_TargetType_PRODUCT_NAME_Target_INDESIGN_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_Product_Target_PhotoshopElement_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_PagenameLevel3_Target_account:licensed-products_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_Domain_Target_account_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_PagenameLevel3_Target_account:plans:switch_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_Domain_Target_stock_adobe_com_COUNT_EventTime_L30',
    'EventType_HAU_MOBILE_TargetType_PRODUCT_NAME_Target_PREMIERE_RUSH_COUNT_EventTime_L30',
    'EventType_HAU_SERVICE_TargetType_PRODUCT_NAME_Target_PORTFOLIO_COUNT_EventTime_L30',
    'EventType_THOR_APP_DOWNLOAD_TargetType_PRODUCT_NAME_Target_Fuse_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_Product_Target_Fuse_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_Domain_Target_behance_net_COUNT_EventTime_L30',
    'EventType_HAU_SERVICE_TargetType_PRODUCT_NAME_Target_NEURAL-FILTERS_COUNT_EventTime_L30',
    'EventType_HAU_WEB_TargetType_PRODUCT_NAME_Target_STOCK_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_PagenameLevel3_Target_account:plans:cancel-reason_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_PagenameLevel3_Target_adobe_com:unsubscribe_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_PagenameLevel3_Target_account:plans:cancel-plan_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_Domain_Target_helpx_adobe_com_COUNT_EventTime_L30',
    'EventType_THOR_APP_DOWNLOAD_TargetType_PRODUCT_NAME_Target_Dreamweaver_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_Product_Target_Spark_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_PagenameLevel3_Target_account:billing-history_COUNT_EventTime_L30',
    'EventType_HAU_DESKTOP_TargetType_PRODUCT_NAME_Target_PHOTOSHOP_COUNT_EventTime_L30',
    'EventType_HAU_MOBILE_TargetType_PRODUCT_NAME_Target_PHOTOSHOP_MIX_COUNT_EventTime_L30',
    'EventType_HAU_DESKTOP_TargetType_PRODUCT_NAME_Target_ILLUSTRATOR_COUNT_EventTime_L30',
    'EventType_HAU_DESKTOP_TargetType_PRODUCT_NAME_Target_AUDITION_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_PagenameLevel3_Target_account:orders_COUNT_EventTime_L30',
    'EventType_HAU_SERVICE_TargetType_PRODUCT_NAME_Target_LEARN_COUNT_EventTime_L30',
    'EventType_HAU_MOBILE_TargetType_PRODUCT_NAME_Target_CAPTURE_COUNT_EventTime_L30',
    'EventType_THOR_APP_DOWNLOAD_TargetType_PRODUCT_NAME_Target_RUSH_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_PagenameLevel3_Target_account:overview_COUNT_EventTime_L30',
    'EventType_THOR_APP_DOWNLOAD_TargetType_PRODUCT_NAME_Target_XD_COUNT_EventTime_L30',
    'EventType_HAU_MOBILE_TargetType_PRODUCT_NAME_Target_ILLUSTRATOR_COUNT_EventTime_L30',
    'EventType_HAU_DESKTOP_TargetType_PRODUCT_NAME_Target_LIGHTROOM_DESKTOP_MAS_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_PagenameLevel3_Target_account:profile_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_Product_Target_LightroomClassic_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_PagenameLevel3_Target_ims:onload_returningusersignedinsuccessfullywithjwt_COUNT_EventTime_L30',
    'EventType_THOR_APP_DOWNLOAD_TargetType_PRODUCT_NAME_Target_Illustrator_COUNT_EventTime_L30',
    'EventType_THOR_APP_DOWNLOAD_TargetType_PRODUCT_NAME_Target_AfterEffects_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_Domain_Target_ims_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_Domain_Target_documentcloud_adobe_com_COUNT_EventTime_L30',
    'EventType_THOR_APP_DOWNLOAD_TargetType_PRODUCT_NAME_Target_PremierePro_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_Product_Target_Behance_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_Product_Target_AfterEffects_COUNT_EventTime_L30',
    'EventType_HAU_MOBILE_TargetType_PRODUCT_NAME_Target_DRAW_COUNT_EventTime_L30',
    'EventType_THOR_APP_DOWNLOAD_TargetType_PRODUCT_NAME_Target_Acrobat_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_Product_Target_Photography_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_PagenameLevel3_Target_account:plans:change-plan_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_Product_Target_XD_COUNT_EventTime_L30',
    'EventType_HAU_MOBILE_TargetType_PRODUCT_NAME_Target_SCAN_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_Domain_Target_adobe_com_COUNT_EventTime_L30',
    'EventType_THOR_APP_DOWNLOAD_TargetType_PRODUCT_NAME_Target_Audition_COUNT_EventTime_L30',
    'EventType_HAU_DESKTOP_TargetType_PRODUCT_NAME_Target_ACROBAT_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_PagenameLevel3_Target_adobe_com:creativecloud:renew_COUNT_EventTime_L30',
    'EventType_HAU_DESKTOP_TargetType_PRODUCT_NAME_Target_DREAMWEAVER_COUNT_EventTime_L30',
    'EventType_HAU_MOBILE_TargetType_PRODUCT_NAME_Target_CC_SURFACE_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_Product_Target_Acrobat_COUNT_EventTime_L30',
    'EventType_HAU_MOBILE_TargetType_PRODUCT_NAME_Target_PHOTOSHOP_CAMERA_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_Domain_Target_downloadcenter_COUNT_EventTime_L30',
    'EventType_HAU_MOBILE_TargetType_PRODUCT_NAME_Target_PHOTOSHOP_EXPRESS_COUNT_EventTime_L30',
    'EventType_WebVisit_TargetType_Product_Target_Stock_COUNT_EventTime_L30',
    'EventType_HAU_MOBILE_TargetType_PRODUCT_NAME_Target_BEHANCE_COUNT_EventTime_L30',
    'EventType_THOR_APP_DOWNLOAD_TargetType_PRODUCT_NAME_Target_Animate_COUNT_EventTime_L30',
    'EventType_HAU_DESKTOP_TargetType_PRODUCT_NAME_Target_BRIDGE_COUNT_EventTime_L30']

LABEL = ['CLTV_30d']

EventTime_L30_FEATURE = NUMERICAL_FEATURE[3:]
BUSINESS_FUNNEL_FACT = {'NON_ACTIVE_USER', 'GROSS_NEW_USER', 'ACTIVE_SUBSCRIBER','PAST_SUBSCRIBER','ACTIVE_USER'}

def get_data(train_dt_range, label_dt_range, sampled_entityID_cnt, business_funnel_line):
  '''
  try to capture users with switched business funnel
  1. GROSS_NEW_USER -> ACTIVE_SUBSCRIBER
  2. ACTIVE_SUBSCRIBER -> PAST_SUBSCRIBER
  3. PAST_SUBSCRIBER -> ACTIVE_SUBSCRIBER
  4. ACTIVE_USER -> ACTIVE_SUBSCRIBER
  5. NON_ACTIVE_USER -> ACTIVE_USER
  6. 
  '''
  initial_funnel = business_funnel_line['initial']
  converted_funnel = business_funnel_line['converted_to']
  assert set(initial_funnel).issubset(BUSINESS_FUNNEL_FACT), 'Invalid initial business funnel'
  assert set(converted_funnel).issubset(BUSINESS_FUNNEL_FACT), 'Invalid converted_to business funnel'
  cols = ID_FEATURE + TIME_FEATURE + NUMERICAL_FEATURE + CATEGORY_FEATURE
  train_base = spark.read.parquet(f'{DATA_PATH}/lstm_training_{train_dt_range[0]}').select(cols).filter(col('business_funnel').isin(initial_funnel))\
    .union(spark.read.parquet(f'{DATA_PATH}/lstm_training_{train_dt_range[1]}').select(cols).filter(col('business_funnel').isin(converted_funnel)))\
    .union(spark.read.parquet(f'{DATA_PATH}/lstm_training_{train_dt_range[2]}').select(cols))\
    .union(spark.read.parquet(f'{DATA_PATH}/lstm_training_{train_dt_range[3]}').select(cols))\
    .union(spark.read.parquet(f'{DATA_PATH}/lstm_training_{train_dt_range[4]}').select(cols))\
    .union(spark.read.parquet(f'{DATA_PATH}/lstm_training_{train_dt_range[5]}').select(cols))\
    .union(spark.read.parquet(f'{DATA_PATH}/lstm_training_{train_dt_range[6]}').select(cols))\
    .union(spark.read.parquet(f'{DATA_PATH}/lstm_training_{train_dt_range[7]}').select(cols))\
    .union(spark.read.parquet(f'{DATA_PATH}/lstm_training_{train_dt_range[8]}').select(cols))\
    .union(spark.read.parquet(f'{DATA_PATH}/lstm_training_{train_dt_range[9]}').select(cols))\
    .union(spark.read.parquet(f'{DATA_PATH}/lstm_training_{train_dt_range[10]}').select(cols))\
    .union(spark.read.parquet(f'{DATA_PATH}/lstm_training_{train_dt_range[11]}').select(cols))
    
  
  """
  load label data (8 months label)
  """
  schema = StructType([\
    StructField('EntityID', StringType(), True),\
    StructField('snapshot_time', TimestampType(), True),\
    StructField('CLTV_30d', DoubleType(), True)])
  label_base = spark.createDataFrame(spark.sparkContext.emptyRDD(), schema)
  for label_dt in label_dt_range:
    label_base = label_base.union(spark.read.parquet(f'{DATA_PATH}/lstm_outcome_{label_dt}').select(ID_FEATURE + TIME_FEATURE + LABEL))
  data_label = label_base.groupBy(ID_FEATURE[0]).pivot('snapshot_time').sum('CLTV_30d').na.fill(0.0)
  """
  rename label columns
  """
  mapping = dict(zip(data_label.columns[1:], ['label_CLTV_'+c[:10] for c in data_label.columns[1:]]))
  data_label = data_label.select([col(c).alias(mapping.get(c, c)) for c in data_label.columns])
  
  train_base = train_base.join(data_label, ['EntityID'],'inner').cache()
  
  """
  get valid eneityID which has 12 months of training data
  """
  print('begin EntityID selection')
  valid_EntityID = train_base.groupBy('EntityID').agg(count('*').alias('cnt')).filter(col('cnt') == 12).select('EntityID').rdd.flatMap(lambda x: x).collect()
  if (len(valid_EntityID) <= sampled_entityID_cnt):
    sampled_entityID_cnt = len(valid_EntityID)
  valid_EntityID = random.sample(valid_EntityID, sampled_entityID_cnt)
  train_EntityID = valid_EntityID[:int(len(valid_EntityID)*0.7)]
  val_EntityID = valid_EntityID[int(len(valid_EntityID)*0.7):int(len(valid_EntityID)*0.8)]
  test_EntityID = valid_EntityID[int(len(valid_EntityID)*0.8):]
  print('complete EntityID selection')
  train_df = train_base.filter(col('EntityID').isin(train_EntityID))
  valid_df = train_base.filter(col('EntityID').isin(val_EntityID))
  test_df = train_base.filter(col('EntityID').isin(test_EntityID))
  
  print(f'Train EntityID cnt: {len(train_EntityID)}')
  print(f'Validation EntityID cnt: {len(val_EntityID)}')
  print(f'Test EntityID cnt: {len(test_EntityID)}')

  return train_df,valid_df,test_df
  

"""
define pipeline: stringIndex categorial columns
"""

country_code_indexer = StringIndexer(inputCol = 'dataset_user_attribute_cc_dc_LATEST_ATTR_country_code', outputCol = 'cat_country', handleInvalid = 'keep')
source_cc_indexer = StringIndexer(inputCol = 'dataset_user_attribute_cc_dc_LATEST_ATTR_cc_signup_source_cc_joining_date_all', outputCol = 'cat_source_cc', handleInvalid = 'keep')
cc_indexer = StringIndexer(inputCol = 'dataset_user_attribute_cc_dc_LATEST_ATTR_cc_signup_for_cc_joining_date_all', outputCol = 'cat_cc', handleInvalid = 'keep')
source_dc_indexer = StringIndexer(inputCol = 'dataset_user_attribute_cc_dc_LATEST_ATTR_dc_signup_source_dc_joining_date_all', outputCol = 'cat_source_dc', handleInvalid = 'keep')
dc_indexer = StringIndexer(inputCol = 'dataset_user_attribute_cc_dc_LATEST_ATTR_dc_signup_for_dc_joining_date_all', outputCol = 'cat_dc', handleInvalid = 'keep')
business_funnel_indexer = StringIndexer(inputCol = 'business_funnel', outputCol = 'cat_business_funnel', handleInvalid = 'keep')


feaurization_stages = [
  country_code_indexer,
  source_cc_indexer,
  cc_indexer,
  source_dc_indexer,
  dc_indexer,
  business_funnel_indexer
]

featurizationPipeLine = Pipeline(stages = feaurization_stages)

'''
minmaxsclae numerial column 
minmaxscale CLTV column need to be done at training stage so that we can do inverse transformation
'''

def minmaxscaler(train_df, numerical_cols, scaler_min, scaler_max):
  aggMin = []
  aggMax = []
  for c in numerical_cols:
    aggMin.append(min(train_df[c]).alias(c))
    aggMax.append(max(train_df[c]).alias(c))

  minlist = train_df.agg(*aggMin).collect()[0]
  maxlist = train_df.agg(*aggMax).collect()[0]
  # transforming ONLY TRAINING data
  for c in numerical_cols:                      
    train_df = train_df.withColumn(c, (train_df[c] - minlist[c])/(maxlist[c]-minlist[c]) * (lit(scaler_max)-lit(scaler_min)) + lit(scaler_min))
  return train_df, minlist, maxlist


def minmaxtransformer(test_df, numerical_cols, minlist, maxlist,scaler_min, scaler_max):
  # transforming the VALID and TEST data (using the training min and max)
  for c in numerical_cols:
    test_df = test_df.withColumn(c, (test_df[c] - minlist[c])/(maxlist[c]-minlist[c]) * (lit(scaler_max)-lit(scaler_min)) + lit(scaler_min))
  return test_df

def get_cols(df):
  """
  get renamed col after data transformation
  """
  cltv_cols = [c for c in df.columns if 'CLTV_' in c]
  cat_cols = [c for c in df.columns if 'cat_' in c]
  ID_FEATURE = ['EntityID']
  TIME_FEATURE = ['snapshot_time']
  # keep original category column for future analysis
  return ID_FEATURE+TIME_FEATURE+cltv_cols+cat_cols+CATEGORY_FEATURE+NUMERICAL_FEATURE[1:]


"""
save data
here we just get active and non-active users
"""
SAVE_PATH = 'dbfs:/mnt/dsnpinternship/rnn-pltv/processed_data' 
def save_data(df, mode, featureCols,business_funnel_switch):
  initial_funnnel = business_funnel_switch['initial'][0]
  converted_funnel = business_funnel_switch['converted_to'][0]
  path = f'{SAVE_PATH}/seq12-label12_v1/{initial_funnnel}_{converted_funnel}/{mode}/'
  df.select(featureCols).write.mode('overwrite').parquet(f'{path}')
  print(f'data saved at {path}')




"""
below is to run code
"""

sampled_entityID_cnt = 120000

business_funnel_switch = {
  'initial':['GROSS_NEW_USER'],
  'converted_to': ['ACTIVE_SUBSCRIBER']
}
train_df,valid_df,test_df = get_data(train_dt_range, label_dt_range,sampled_entityID_cnt,business_funnel_switch)
featureizationPipeLine_fit = featurizationPipeLine.fit(train_df)

train_df_transformed = featureizationPipeLine_fit.transform(train_df)

val_df_transformed = featureizationPipeLine_fit.transform(valid_df)
test_df_transformed = featureizationPipeLine_fit.transform(test_df)

col_selected = get_cols(val_df_transformed)
train_df_transformed, minlist, maxlist = minmaxscaler(train_df_transformed, NUMERICAL_FEATURE[1:], 0, 1)
val_df_transformed = minmaxtransformer(val_df_transformed, NUMERICAL_FEATURE[1:], minlist, maxlist,0, 1)
test_df_transformed = minmaxtransformer(test_df_transformed, NUMERICAL_FEATURE[1:], minlist, maxlist,0, 1)

save_data(train_df_transformed, 'train',col_selected)
save_data(val_df_transformed, 'val',col_selected)
save_data(test_df_transformed, 'test',col_selected)
