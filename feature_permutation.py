def _mse(y_true, y_pred):
  return metrics.mean_squared_error(y_true, y_pred, multioutput='raw_values')
def _rmse(y_true, y_pred):
  return metrics.mean_squared_error(y_true, y_pred, squared = False,multioutput='raw_values')
def _avg_abs_error(y_true, y_pred):
  return metrics.mean_absolute_error(y_true, y_pred, multioutput='raw_values')


def feature_importance(model_loaded):
  """
  The permutation feature importance algorithm based on Fisher, Rudin, and Dominici (2018)
  """
  with torch.no_grad():      
      cnt = 0
      feature_importance_col = ['mse','rmse','mae']
      num_feature_importance = np.array([[0.0 for _ in range(len(cltv_Cols_train+numerial_Cols))] for _ in range(len(feature_importance_col))])
      cat_feature_importance = np.array([[0.0 for _ in range(len(CATEGORY_FEATURE))] for _ in range(len(feature_importance_col))])

      for cat_x_test, num_x_test, y_test in test_loader_one_shuffled:
        cnt += 1
        # only use 10000 sample
        if (cnt > 10000):
          break
        cat_x_test = cat_x_test.to(device)
        num_x_test = num_x_test.to(device)
        y_test = y_test.to(device)

        model_loaded.eval()
        yhat = model_loaded(cat_x_test,num_x_test)
        mse_error_orig = _mse(yhat.cpu().detach().numpy(),y_test.cpu().detach().numpy())
        rmse_error_orig = _rmse(yhat.cpu().detach().numpy(),y_test.cpu().detach().numpy())
        mae_error_orig = _avg_abs_error(yhat.cpu().detach().numpy(),y_test.cpu().detach().numpy())
          
        for i in range(num_x_test.shape[2]):
          num_x_test_perbutated =  num_x_test.clone().detach()
          #print('before',num_x_test_perbutated[:,:,i])
          num_x_test_perbutated[:,:,i] = torch.HalfTensor(np.random.normal(0.5, 0.1, num_x_test_perbutated.shape[0]))
          yhat_perbutated = model_loaded(cat_x_test,num_x_test_perbutated)  

          mse_error_perm = _mse(yhat_perbutated.cpu().detach().numpy(),y_test.cpu().detach().numpy())
          rmse_error_perm  = _rmse(yhat_perbutated.cpu().detach().numpy(),y_test.cpu().detach().numpy())
          mae_error_perm  = _avg_abs_error(yhat_perbutated.cpu().detach().numpy(),y_test.cpu().detach().numpy())
          
          num_feature_importance[0][i] += _avg_abs_error(mse_error_perm, mse_error_orig)
          num_feature_importance[1][i] += _avg_abs_error(rmse_error_perm, rmse_error_orig)
          num_feature_importance[2][i] += _avg_abs_error(mae_error_perm, mae_error_orig)
          #num_x_test_perbutated[:,:,i]= num_x_test_perbutated[:,:,i].normal_(mean = 0.5, std = 0.1)
          #print('after',num_x_test_perbutated[:,:,i])
          #print('all',num_x_test_perbutated)
          
        cat_feaute_name = list(EMBEDDING_SIZES.keys())
        for j in range(cat_x_test.shape[2]):
          cat_x_test_perbutated =  cat_x_test.clone().detach()
          if (cnt %2500 ==0):
            print(f'cnt = {cnt}, before',cat_x_test_perbutated[:,:,j])
          #cat_x_test_perbutated[:,:,i] = torch.HalfTensor(np.random.normal(0.5, 0.1, num_x_test_perbutated.shape[0]))
          cat_x_test_perbutated[:,:,j] = torch.tensor([random.randint(0, EMBEDDING_SIZES[cat_feaute_name[j]][0]-1) for _ in range(cat_x_test_perbutated.shape[0])]) 
          if (cnt %2500 ==0):
            print(f'cnt = {cnt},  after',cat_x_test_perbutated[:,:,j])
            print('==================================')
          yhat_perbutated = model_loaded(cat_x_test_perbutated,num_x_test)

          mse_error_perm = _mse(yhat_perbutated.cpu().detach().numpy(),y_test.cpu().detach().numpy())
          rmse_error_perm = _rmse(yhat_perbutated.cpu().detach().numpy(),y_test.cpu().detach().numpy())
          mae_error_perm = _avg_abs_error(yhat_perbutated.cpu().detach().numpy(),y_test.cpu().detach().numpy())

          cat_feature_importance[0][j] += _avg_abs_error(mse_error_perm, mse_error_orig)
          cat_feature_importance[1][j] += _avg_abs_error(rmse_error_perm, rmse_error_orig)
          cat_feature_importance[2][j] += _avg_abs_error(mae_error_perm, mae_error_orig)
  feature_importance_df = pd.concat([pd.DataFrame(data = np.transpose(cat_feature_importance), index = CATEGORY_FEATURE, columns = feature_importance_col)\
    ,pd.DataFrame(data = np.transpose(num_feature_importance), index = cltv_Cols_train+numerial_Cols,columns = feature_importance_col)])\
    .rename_axis('feature').reset_index().sort_values(by=['mse'],ascending=False)
  return feature_importance_df
      
      