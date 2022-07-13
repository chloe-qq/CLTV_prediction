

def Train(model,train, val, test,model_num):
  print(f'-------------- training model {model_num} ----------------------')
  train_loader = DataLoader(train, batch_size=para['batch_size'], shuffle=True, drop_last=False)
  val_loader = DataLoader(val, batch_size=para['batch_size'], shuffle=True, drop_last=False)
  test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=False)
  
  loss_fn = nn.MSELoss(reduction="mean")
  optimizer = optim.AdamW(model.parameters(), lr=para['learning_rate'],weight_decay = 0.00001)
  opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer, patience = 3 )
  
  print(summary(model))
  
  opt.train(train_loader, 
          val_loader, 
          batch_size= para['batch_size'], 
          n_epochs=para['max_epochs'])
  opt.plot_losses()
  
  predictions, values = opt.evaluate(test_loader_one, batch_size=1)
  print(f'-------------- testing model {model_num} ----------------------')
  pred_raw,label_raw = inverse_scale(predictions,values,maxlist_dict, minlist_dict,labelCols)
  return model,pred_raw, label_raw

"""
business funnel result
"""
def business_funnel_result(pred_raw, label_raw,test_fact_df):  
  # signup_source = 'cc','dc'
  test_result_raw,cc_source_level_statistis,user_funnel_statistis,pred_label_df = loss_eval(pred_raw, label_raw, labelCols, test_fact_df, 'cc', if_divided_by_business_funnel=True)
  _,dc_source_level_statistis,_,_ = loss_eval(pred_raw, label_raw, labelCols, test_fact_df.reset_index(), 'dc', if_divided_by_business_funnel=True)
  return test_result_raw,user_funnel_statistis,cc_source_level_statistis,dc_source_level_statistis

