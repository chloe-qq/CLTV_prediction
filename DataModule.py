class DataModule():
  def __init__(self, seq_length, numericalCols, categorialCols,CATEGORY_FEATURE, labelCols, data_path, cat_feature_dict):
    super().__init__()
    self.seq_length = seq_length
    self.numericalCols = numericalCols
    self.categorialCols = categorialCols
    self.CATEGORY_FEATURE = CATEGORY_FEATURE
    self.labelCols = labelCols
    self.data_path = data_path
    self.cat_feature_dict = cat_feature_dict
    
    self.cltv_Cols_train = ['CLTV_30d']
    self.fact_Cols = ['EntityID','snapshot_time']
    self.AllNumericalCols = self.cltv_Cols_train + self.numericalCols

    #idx_cat_Cols = ['cat_country','cat_source_cc','cat_cc','cat_source_dc','cat_dc','cat_business_funnel']
    cat_makeup_i = [0 for _ in range(len(self.categorialCols))]
    for i in range(len(categorialCols)):
      # the length of each categorial feature indicates the null value
      # set all categorial features to be null if a user has not registered
      cat_makeup_i[i] = self.cat_feature_dict[self.categorialCols[i]]
    self.cat_makeup = cat_makeup_i
    self.num_makeup = [0.0 for _ in range(len(self.AllNumericalCols))]

    """
     data 12 months preduct 12 months
    """    
    
  def minmaxscaler(self, train_df, scaler_min, scaler_max):
    cols = self.AllNumericalCols + self.labelCols
    aggMin = []
    aggMax = []
    for c in cols:
      aggMin.append(min(train_df[c]).alias(c))
      aggMax.append(max(train_df[c]).alias(c))

    minlist = train_df.agg(*aggMin).collect()[0]
    maxlist = train_df.agg(*aggMax).collect()[0]
    # transforming the TRAINING data
    for c in self.AllNumericalCols:                      
      train_df = train_df.withColumn(c, (train_df[c] - minlist[c])/(maxlist[c]-minlist[c]) * (lit(scaler_max)-lit(scaler_min)) + lit(scaler_min))
    for c in self.labelCols:                      
      train_df = train_df.withColumn('scaled_'+c, (train_df[c] - minlist[c])/(maxlist[c]-minlist[c]) * (lit(scaler_max)-lit(scaler_min)) + lit(scaler_min))  
      
    return train_df, minlist, maxlist
  
  def minmaxtransformer(self, test_df, minlist, maxlist,scaler_min, scaler_max):
    # transforming the VALID and TEST data (using the training min and max)
    for c in self.AllNumericalCols:
      test_df = test_df.withColumn(c, (test_df[c] - minlist[c])/(maxlist[c]-minlist[c]) * (lit(scaler_max)-lit(scaler_min)) + lit(scaler_min))
    for c in self.labelCols:                      
      test_df = test_df.withColumn('scaled_'+c, (test_df[c] - minlist[c])/(maxlist[c]-minlist[c]) * (lit(scaler_max)-lit(scaler_min)) + lit(scaler_min)) 
    return test_df
  
  
  def impute_nan(self, df):
    # as EventType_WebVisit_TargetType_PagenameLevel3_Target_account:activated-devices_COUNT_EventTime_L30 column will lead to null as max - min == 0
    # in case other column will have the same situation
    df = df.fillna(0.0, subset= self.numericalCols + self.labelCols + ['scaled_'+c for c in self.labelCols])
    df = df.fillna('UNKNOWN', subset= self.CATEGORY_FEATURE)
    return df
  
  
  def load_parquet(self,mode):
    df = spark.read.parquet(f'{self.data_path}/{mode}/*').dropDuplicates()
    # do orderBy at creating sequence
    
    EntityID_list = df.select('EntityID').distinct().rdd.flatMap(lambda x: x).collect()    
    print(f'loading {mode} data, EntityID count = {len(EntityID_list)} ')
    return df, EntityID_list
  
  def get_df(self):
    df_train, train_EntityID_list  = self.load_parquet('train')
    df_val, val_EntityID_list =  self.load_parquet('val')
    df_train, minlist, maxlist = self.minmaxscaler(df_train, 0, 1)
    df_val = self.minmaxtransformer(df_val, minlist, maxlist, 0,1)
    df_train = self.impute_nan(df_train)
    df_val= self.impute_nan(df_val)
    return df_train,train_EntityID_list, df_val,val_EntityID_list, minlist, maxlist 
   
  
  def create_sequence(self,input_data, if_test_dataset):    
    num_features = []
    cat_features = []
    labels = []
  
    schema_categorial = []
    for c in CATEGORY_FEATURE:
      schema_categorial.append(StructField(c, StringType(), True))
    schema = StructType([\
      StructField('EntityID', StringType(), False),\
      StructField('snapshot_time', TimestampType(), False)] + schema_categorial\
      )
    fact_df = spark.createDataFrame(spark.sparkContext.emptyRDD(), schema)

    labelCols = ['scaled_'+ c for c in self.labelCols]
    
    # get seq length info 
    seq_window = Window.partitionBy("entityID")
    input_data  = input_data.withColumn("cur_seq_length", count("snapshot_time").over(seq_window))
    throw_active_sub_cnt = 0
    for seq in range(self.seq_length,0,-1):
      # select entityID whose seq_length = i
      print(f'processing EntityID with exsiting seq-length = {seq}')
      fact_position = []
      df_i = input_data.filter(col('cur_seq_length')==seq).drop('cur_seq_length').orderBy('EntityID','snapshot_time').toPandas()
      
      data_size = len(df_i)
      stride = seq
      for i in range(0, data_size - seq +1, stride):
        # throw away users with last snapshot as active subscriber, cat_businss_funnel == 2
        cat_feature_i = np.array(df_i[i:i+stride][self.categorialCols])
        if (cat_feature_i[-1][5] == 2):
          throw_active_sub_cnt += 1
          continue
        num_feature_i = np.array(df_i[i:i+stride][self.AllNumericalCols]).astype('float')
        

        label_position = i+stride-1
        fact_position.append(label_position)
        label_i = np.array(df_i.iloc[label_position][labelCols]).astype('float')
        if (seq < self.seq_length):
          num_makeup_list = np.array([self.num_makeup for _ in range(self.seq_length - seq)]).astype('float')
          num_feature_i = np.append(num_makeup_list,num_feature_i,axis = 0)
          
          # for catgorial features
          # check if gn-user
          if (cat_feature_i[0][5] == 4):
            cat_makeup_list = np.array([self.cat_makeup for i in range(self.seq_length - seq)])
            cat_feature_i = np.append(cat_makeup_list,cat_feature_i,axis = 0)
          else:
            # switched to non-active user
            # also need to REVISE the fact df to make the funnel to be non-active user
            
            if (seq <=9 and if_test_dataset == False):
              # if inactive month >=6, throw away that user
              fact_position.pop(-1)
              continue
            else:
              df_i.loc[label_position,'business_funnel'] = 'NON_ACTIVE_USER'

              cat_makeup_ = cat_feature_i[0]
              # set user funnel to be non-active user (0), other categorial columns are the same as the current one.
              cat_makeup_[5] = 0
              cat_makeup_list = np.array([cat_makeup_ for i in range(self.seq_length - seq)])
              # reverse the order of append
              cat_feature_i = np.append(cat_feature_i, cat_makeup_list, axis = 0)

        assert len(num_feature_i) == self.seq_length,f'numerical feature sequence length shoud be {self.seq_length}'
        assert len(cat_feature_i) == self.seq_length,f'categoical feature sequence length shoud be {self.seq_length}'
        num_features.append(num_feature_i)
        cat_features.append(cat_feature_i)
        labels.append(label_i)
      if(seq == self.seq_length):
        fact_df = df_i.iloc[fact_position][self.fact_Cols + self.CATEGORY_FEATURE]
      else:
        fact_df_i = df_i.iloc[fact_position][self.fact_Cols + self.CATEGORY_FEATURE]
        fact_df = pd.concat([fact_df, fact_df_i], ignore_index=True)
    print(f'thrown away active-sub(last snap-shot) count = {throw_active_sub_cnt}')
    return cat_features,num_features,labels, fact_df
  
  def get_tensorsDataset(self, df):    
    cat_features,num_features,labels, fact_df = self.create_sequence(df)
    cat_features_tensor = torch.Tensor(cat_features)
    num_features_tensor = torch.Tensor(num_features)
    label_tensor = torch.Tensor(labels)
    
    return TensorDataset(cat_features_tensor, num_features_tensor,label_tensor), fact_df
  
  def get_tensorDataset_from_pt(self,loading_path,mode, if_exclude_cur_cltv, if_exclude_active_sub):
    if (mode == 'test' or if_exclude_active_sub == False):
      print('for testing data, use the same (include active subscriber)')
      path_cat_feature = f'{loading_path}/sequence_array/{mode}_cat_features_tensor.pt'
      path_num_feature = f'{loading_path}/sequence_array/{mode}_num_features_tensor.pt'
      path_label = f'{loading_path}/sequence_array/{mode}_labels_tensor.pt'
      path_fact_table = f'{loading_path}/fact_table/{mode}_fact.csv'
    else:
      path_cat_feature = f'{loading_path}/sequence_array_v2/{mode}_cat_features_tensor.pt'
      path_num_feature = f'{loading_path}/sequence_array_v2/{mode}_num_features_tensor.pt'
      path_label = f'{loading_path}/sequence_array_v2/{mode}_labels_tensor.pt'
      path_fact_table = f'{loading_path}/fact_table_v2/{mode}_fact.csv'
    cat_features_tensor = torch.load(path_cat_feature)
    num_features_tensor = torch.load(path_num_feature)
    label_tensor = torch.load(path_label)
    fact_table = pd.read_csv(path_fact_table)
    assert len(label_tensor) == len(fact_table),'FACT TABLE SIZE MUST EQUAL TO LABEL LENGTH'
    if (if_exclude_cur_cltv):
      num_features_tensor = num_features_tensor[:,:,1:]
    return TensorDataset(cat_features_tensor, num_features_tensor,label_tensor), fact_table
