

def __mse(y_true, y_pred):
      return metrics.mean_squared_error(y_true, y_pred, multioutput='raw_values')
def __rmse(y_true, y_pred):
  return metrics.mean_squared_error(y_true, y_pred, squared = False,multioutput='raw_values')
def __avg_abs_error(y_true, y_pred):
  return metrics.mean_absolute_error(y_true, y_pred, multioutput='raw_values')

def get_3_error(yhat, y_test):
  mse_error_orig = __mse(yhat,y_test)
  rmse_error_orig = __rmse(yhat,y_test)
  mae_error_orig = __avg_abs_error(yhat,y_test)
  return mse_error_orig, rmse_error_orig, mae_error_orig

def _update(model_name, yhat_perbutated, y_test, mse_error_orig,rmse_error_orig, mae_error_orig,i,feature_importance):
  mse_error_perm, rmse_error_perm, mae_error_perm = get_3_error(yhat_perbutated.cpu().detach().numpy(),y_test.cpu().detach().numpy())
  # take difference of the two error
  feature_importance[model_name][0][i] += __avg_abs_error(mse_error_perm, mse_error_orig)
  feature_importance[model_name][1][i] += __avg_abs_error(rmse_error_perm, rmse_error_orig)
  feature_importance[model_name][2][i] += __avg_abs_error(mae_error_perm, mae_error_orig)


def feature_importance(sample_cnt,only_active_sub, exclude_active_sub):
  """
  The permutation feature importance algorithm based on Fisher, Rudin, and Dominici (2018)
  calculate feature importance of lstm-cnn-embedding model of 3 versions (12-month, 6-month, 3-month)
  """    
  test_loader_one = DataLoader(test_1, batch_size=1, shuffle=True, drop_last=False)
  model_lstm_cnn_embedding_3month = load_model(model_name = 'lstm_cnn_embedding_3month').eval()
  model_lstm_cnn_embedding_6month = load_model(model_name = 'lstm_cnn_embedding_6month').eval()
  model_lstm_cnn_embedding_12month = load_model(model_name = 'lstm_cnn_embedding').eval()

  assert only_active_sub!=exclude_active_sub, 'only_active_sub and exclude_active_sub MUST BE OPPOSITE'
  if (only_active_sub):
    sample_cnt = sample_cnt/10
    print(f'for only_active_sub, update sample_cnt = {sample_cnt}')
  with torch.no_grad():      
      cnt = 0
      skip_user_cnt = 0
      feature_importance_col = ['mse','rmse','mae']
      
      num_feature_importance_template = np.array([[0.0 for _ in range(len(cltv_Cols_train+numerial_Cols))] for _ in range(len(feature_importance_col))])
      cat_feature_importance_template = np.array([[0.0 for _ in range(len(CATEGORY_FEATURE))] for _ in range(len(feature_importance_col))])
      # NEED TO ADD .copy() here, or all the array will be revised
      num_feature_importance = {'lstm_cnn_12month': num_feature_importance_template.copy(), 
                               'lstm_cnn_6month': num_feature_importance_template.copy(),
                               'lstm_cnn_3month': num_feature_importance_template.copy()}
      cat_feature_importance = {'lstm_cnn_12month': cat_feature_importance_template.copy(), 
                               'lstm_cnn_6month': cat_feature_importance_template.copy(),
                               'lstm_cnn_3month': cat_feature_importance_template.copy()}
      model_list = ['lstm_cnn_12month','lstm_cnn_6month','lstm_cnn_3month']
      feature_importance_df = {}
      
      for cat_x_test, num_x_test, y_test in test_loader_one:
        # exclude active_subscriber when calculating feature importance, ('NON_ACTIVE_USER', 'ACTIVE_USER', 'ACTIVE_SUBSCRIBER', 'PAST_SUBSCRIBER', 'GROSS_NEW_USER')
        if (exclude_active_sub and cat_x_test[0][-1][5] == 2):
          skip_user_cnt += 1
          continue
        elif (only_active_sub):
          if (cat_x_test[0][-1][5] != 2):
            continue
        # only use 20000 sample
        if (cnt >= sample_cnt):
          break
        cnt += 1
        cat_x_test = cat_x_test.to(device)
        num_x_test = num_x_test.to(device)
        y_test = y_test.to(device)
        
        yhat_12 = model_lstm_cnn_embedding_12month(cat_x_test,num_x_test)
        yhat_6 = model_lstm_cnn_embedding_6month(cat_x_test,num_x_test)
        yhat_3 = model_lstm_cnn_embedding_3month(cat_x_test,num_x_test)
        mse_error_orig_12,rmse_error_orig_12, mae_error_orig_12  = get_3_error(yhat_12.cpu().detach().numpy(),y_test.cpu().detach().numpy())
        mse_error_orig_6,rmse_error_orig_6, mae_error_orig_6  = get_3_error(yhat_6.cpu().detach().numpy(),y_test.cpu().detach().numpy())
        mse_error_orig_3,rmse_error_orig_3, mae_error_orig_3  = get_3_error(yhat_3.cpu().detach().numpy(),y_test.cpu().detach().numpy())
        
          
        for i in range(num_x_test.shape[2]):
          num_x_test_perbutated =  num_x_test.clone().detach()
          #print('before',num_x_test_perbutated[:,:,i])
          num_x_test_perbutated[:,:,i] = torch.HalfTensor(np.random.normal(0.5, 0.1, num_x_test_perbutated.shape[0]))
          yhat_perbutated_12 = model_lstm_cnn_embedding_12month(cat_x_test,num_x_test_perbutated)  
          yhat_perbutated_6 = model_lstm_cnn_embedding_6month(cat_x_test,num_x_test_perbutated)
          yhat_perbutated_3 = model_lstm_cnn_embedding_3month(cat_x_test,num_x_test_perbutated)
          
          _update('lstm_cnn_12month', yhat_perbutated_12, y_test, mse_error_orig_12,rmse_error_orig_12, mae_error_orig_12,i,num_feature_importance)
          _update('lstm_cnn_6month', yhat_perbutated_6, y_test, mse_error_orig_6,rmse_error_orig_6, mae_error_orig_6,i,num_feature_importance)
          _update('lstm_cnn_3month', yhat_perbutated_3, y_test, mse_error_orig_3,rmse_error_orig_3, mae_error_orig_3,i,num_feature_importance)

        cat_feaute_name = list(EMBEDDING_SIZES.keys())
        for j in range(cat_x_test.shape[2]):
          cat_x_test_perbutated =  cat_x_test.clone().detach()
          if (cnt %10000 ==0 and j == 0):
            print(f'cnt = {cnt}, before',cat_x_test_perbutated[:,:,j])
          cat_x_test_perbutated[:,:,j] = torch.tensor([random.randint(0, EMBEDDING_SIZES[cat_feaute_name[j]][0]-1) for _ in range(cat_x_test_perbutated.shape[0])]) 
          if (cnt %10000 ==0 and j == 0):
            print(f'cnt = {cnt},  after',cat_x_test_perbutated[:,:,j])
            print('===========================================')
          yhat_perbutated_12 = model_lstm_cnn_embedding_12month(cat_x_test_perbutated,num_x_test)
          yhat_perbutated_6 = model_lstm_cnn_embedding_6month(cat_x_test_perbutated,num_x_test)
          yhat_perbutated_3 = model_lstm_cnn_embedding_3month(cat_x_test_perbutated,num_x_test)
          
          _update('lstm_cnn_12month', yhat_perbutated_12, y_test, mse_error_orig_12,rmse_error_orig_12, mae_error_orig_12,j,cat_feature_importance)
          _update('lstm_cnn_6month', yhat_perbutated_6, y_test, mse_error_orig_6,rmse_error_orig_6, mae_error_orig_6,j,cat_feature_importance)
          _update('lstm_cnn_3month', yhat_perbutated_3, y_test, mse_error_orig_3,rmse_error_orig_3, mae_error_orig_3,j,cat_feature_importance)
      for k in model_list:
        cat_feature_importance[k] = cat_feature_importance[k]/sample_cnt
        num_feature_importance[k] = num_feature_importance[k]/sample_cnt
        feature_importance_df[k] = pd.concat([pd.DataFrame(data = np.transpose(cat_feature_importance[k]), index = CATEGORY_FEATURE, columns = feature_importance_col)\
          ,pd.DataFrame(data = np.transpose(num_feature_importance[k]), index = cltv_Cols_train+numerial_Cols,columns = feature_importance_col)])\
          .rename_axis('feature').reset_index().sort_values(by=['rmse'],ascending=False)
  if (exclude_active_sub):
    print(f'skip active_subscriber count: {skip_user_cnt}')
  return feature_importance_df