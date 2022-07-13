
def inverse_scale(pred_,label_,maxlist,minlist,labelCols):
  """
  rescale the cltv to the origin values
  yhat_original = yhat*(max-min)+min
  """
  pred = copy.deepcopy(pred_)
  label = copy.deepcopy(label_)
  for m, month in enumerate(labelCols):
    for batch in pred:
      for item in batch:
        item[m] =item[m]*(maxlist[month]-minlist[month])+minlist[month]
        
    for batch in label:
      for item in batch:
        item[m] =item[m]*(maxlist[month]-minlist[month])+minlist[month]

  return pred,label


# https://github.com/google/lifetime_value/blob/master/lifetime_value/metrics.py

def _normalized_rmse(y_true, y_pred):
  return np.sqrt(metrics.mean_squared_error(y_true, y_pred)) / y_true.mean()

def _normalized_mae(y_true, y_pred):
  return metrics.mean_absolute_error(y_true, y_pred) / y_true.mean()

def _rmse(y_true, y_pred):
  return np.sqrt(metrics.mean_squared_error(y_true, y_pred))

def _total_abs_error(y_true, y_pred):
  return metrics.mean_absolute_error(y_true, y_pred)* len(y_true)

def _avg_abs_error(y_true, y_pred):
  return metrics.mean_absolute_error(y_true, y_pred)

def _mape(y_true, y_pred):
  return np.abs(np.mean(y_pred) - np.mean(y_true)) / np.mean(y_true)

def _total_percentage_error(y_true, y_pred):
  return np.abs(np.sum(y_pred) - np.sum(y_true)) / np.sum(y_true)

def _aggregate_fn(df):
  return pd.Series({
      'label_sum': np.sum(df['y_true']),
      'pred_sum': np.sum(df['y_pred']),
      'total_error_percentage':_total_percentage_error(df['y_true'], df['y_pred']),
      'total_abs_error': _total_abs_error(df['y_true'], df['y_pred']),
      'rmse': _rmse(df['y_true'], df['y_pred']),
      'avg_abs_error': _avg_abs_error(df['y_true'], df['y_pred']),
      'mape': _mape(df['y_true'], df['y_pred']),
      'normalized_rmse': _normalized_rmse(df['y_true'], df['y_pred']),
      'normalized_mae': _normalized_mae(df['y_true'], df['y_pred'])
  })

  
def business_funnel_stats(y_true, y_pred, test_fact_df) -> pd.DataFrame:
  """
  aggreate by business funnel
  """

  df = pd.DataFrame({
      'y_true': y_true,
      'y_pred': y_pred,
      f'business_funnel': np.array(test_fact_df['business_funnel'])
  }).groupby('business_funnel').apply(_aggregate_fn)

  return df.reset_index()

def cc_dc_source_stats(y_true, y_pred, test_fact_df, source_name, if_divided_by_business_funnel) -> pd.DataFrame:
  """
  aggreate by signup source
  """

  if (if_divided_by_business_funnel):
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        f'business_funnel': np.array(test_fact_df['business_funnel']),
        f'{source_name}': np.array(test_fact_df[source_name])
    }).groupby(['business_funnel', f'{source_name}']).apply(_aggregate_fn)
  else:
    df = pd.DataFrame({
      'y_true': y_true,
      'y_pred': y_pred,
      f'signup_source_{source_name}': np.array(test_fact_df[source_name])
  }).groupby(f'signup_source_{source_name}').apply(_aggregate_fn)

  return df.reset_index()



def decile_stats(y_true, y_pred, month,num_buckets) -> pd.DataFrame:
  """Calculates decile level means and errors.
  The function first partites the examples into ten equal sized
  buckets based on sorted `y_pred`, and computes aggregated metrics in each
  bucket.
  Arguments:
    y_true: True labels.
    y_pred: Predicted labels.
  Returns:
    df: Bucket level statistics.
    aggregate level is to set num_bucket = 1
  """

  #decile = pd.qcut(y_pred, q=num_buckets, labels=['%d' % i for i in range(num_buckets)], duplicates = 'raise', retbins = True)
  decile = pd.qcut(y_pred, q=num_buckets, duplicates = 'drop')

  df = pd.DataFrame({
      'y_true': y_true,
      'y_pred': y_pred,
      'decile': decile,
  }).groupby('decile').apply(_aggregate_fn)

  return df



def loss_eval(pred, true, labelCols, test_fact_df, signup_source, if_divided_by_business_funnel):
  """
  calculate the aggregated level mse loss predictions and label
  level: 
    dc_signup_source: DESKTOP, MOBILE, WEB
    cc_signup_source: DESKTOP, MOBILE
  """
  if (signup_source == 'dc'):
    source_name = 'dataset_user_attribute_cc_dc_LATEST_ATTR_dc_signup_source_dc_joining_date_all'
  if (signup_source == 'cc'):
    source_name = 'dataset_user_attribute_cc_dc_LATEST_ATTR_cc_signup_source_cc_joining_date_all'
  pred_cltv = {}
  label_cltv = {}
  total_mpe = {}
  total_true_cltv = {}
  total_pred_cltv = {}
  mse_test_loss = {}
  mae_test_loss = {}
  sum_mae_test_loss = {}
  
  source_level_statistis = {}
  gn_user_statistis = {}
    
  for m,month in enumerate(labelCols):
    pred_cltv_i = []
    label_cltv_i = []
    
    for batch in pred:
      for item in batch:
        pred_cltv_i.append(item[m])
    pred_cltv_i = np.array(pred_cltv_i)
    pred_cltv['pred_'+month[6:]] = pred_cltv_i
    
    for batch_ in true:
      for item_ in batch_:
        label_cltv_i.append(item_[m])
    label_cltv_i = np.array(label_cltv_i)
    label_cltv['true_'+month[6:]] = label_cltv_i
    
    total_true_cltv[month] = np.sum(label_cltv_i)
    total_pred_cltv[month] = np.sum(pred_cltv_i)
    total_mpe[month] = np.abs(total_true_cltv[month]-total_pred_cltv[month])/total_true_cltv[month]
    mse_test_loss[month] = np.mean((pred_cltv_i-label_cltv_i)**2)
    mae_test_loss[month] = np.mean(np.abs(pred_cltv_i-label_cltv_i))
    sum_mae_test_loss[month] = np.sum(np.abs(pred_cltv_i-label_cltv_i))
    
    assert source_name is not None, 'Not assign source name!'
    source_level_statistis[month] = cc_dc_source_stats(label_cltv_i, pred_cltv_i, test_fact_df, source_name, if_divided_by_business_funnel)
    gn_user_statistis[month] = business_funnel_stats(label_cltv_i, pred_cltv_i, test_fact_df)
    
  total_cltv_true_df = pd.DataFrame.from_dict(total_true_cltv, orient='index',columns = ['sum_true_cltv'])
  total_cltv_pred_df = pd.DataFrame.from_dict(total_pred_cltv, orient='index',columns = ['sum_pred_cltv'])
  total_mpe_df = pd.DataFrame.from_dict(total_mpe, orient='index',columns = ['total_percentage_error'])
  mse_loss_df = pd.DataFrame.from_dict(mse_test_loss, orient='index',columns = ['mse'])
  mae_loss_df = pd.DataFrame.from_dict(mae_test_loss, orient='index',columns = ['mae'])
  mae_total_loss_df = pd.DataFrame.from_dict(sum_mae_test_loss, orient='index',columns = ['total_abs_error'])
    
  total_statistic = total_cltv_true_df.merge(total_cltv_pred_df, left_index = True, right_index = True)\
    .merge(total_mpe_df, left_index = True, right_index = True)\
    .merge(mse_loss_df, left_index = True, right_index = True)\
    .merge(mae_loss_df,  left_index = True, right_index = True)\
          .merge(mae_total_loss_df,  left_index = True, right_index = True)
  
  pred_label_df = (pd.DataFrame.from_dict(pred_cltv, orient='index').T).merge(pd.DataFrame.from_dict(label_cltv, orient='index').T, left_index = True, right_index = True)

  return total_statistic,source_level_statistis, gn_user_statistis,pred_label_df


