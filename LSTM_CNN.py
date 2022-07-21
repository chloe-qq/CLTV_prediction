"""
reference
https://arxiv.org/abs/1703.07015
https://github.com/laiguokun/LSTNet


LSTM + CNN + embedding
no skip, no highway
"""

model_params_lstm_cnn_embedding = {
                'seq_length':12,
                'num_numcols': len(numerial_Cols+cltv_Cols_train),
                'embedding_sizes' : EMBEDDING_SIZES,
                'x_categoricals' : idx_cat_Cols,
                'categorical_groups':[],
                'dropout':0.2,
                'output_dim' : len(labelCols),
                'hidden_dim': 35, #cnn out = 35, 35+13(categorical feature dim after embedding)=48 feed into lstm
                'num_layers':1,
                'threshold': -0.6,
                'flap_coefficient': -0.001
               }


class LSTM_CNN(nn.Module):
  def __init__(self, 
              seq_length, 
              num_dim,
              emb_out_dim,
              output_dim,
              hidRNN,
              hidCNN,
              cnn_kernel_size,
              attn,
              highway_window,
              dropout):
    super(LSTM_CNN, self).__init__()
    self.seq_length = seq_length
    self.num_dim = num_dim
    self.emb_out_dim = emb_out_dim
    self.output_dim = output_dim
    self.hidRNN = hidRNN
    self.hidCNN = hidCNN
    self.cnn_kernel_size = cnn_kernel_size
    self.highway_window = highway_window
    self.attn = attn
    self.dropout = dropout 

    # CNN convulation, where cnn_kernel_size decide how to shrink the timestep, shirnked_time_step = seq_length - cnn_kernel_size + 1
    self.conv1 = nn.Conv2d(in_channels = 1, out_channels = self.hidCNN, kernel_size = (self.cnn_kernel_size, self.num_dim))

    # one layer GRU or LSTM, no dropout in RNN cell
    #self.gru = nn.GRU(input_size  = self.hidCNN+self.emb_out_dim, hidden_size = self.hidRNN, batch_first = True)
    self.lstm = nn.LSTM(input_size = self.hidCNN+self.emb_out_dim, hidden_size = self.hidRNN, batch_first = True)
    self.dropout_layer = nn.Dropout(p = self.dropout)
    
  
    self.multihead = None
    if (self.attn):
        self.multihead = nn.MultiheadAttention(embed_dim=self.hidRNN, num_heads=3)
        self.fc = nn.Linear(in_features=self.hidRNN*2, out_features=self.output_dim)    
    else:
        self.fc = nn.Linear(self.hidRNN, self.output_dim);
    if (self.highway_window > 0):
        self.highway = nn.Linear(self.highway_window, self.output_dim);

  def forward(self, cat_x_embbed, num_x):
      batch_size = cat_x_embbed.size(0)
      assert batch_size == num_x.size(0),'num features size is not equal to cat feature size'

      # CNN
      num_x = num_x.view(-1, 1, self.seq_length, self.num_dim)
      num_x = F.relu(self.conv1(num_x))
      num_x = self.dropout_layer(num_x);
      
      num_x = torch.squeeze(num_x, 3); # [batch_size, conv1_out_channels, shrinked_timesetps]
      num_x = num_x.permute(0,2,1).contiguous() # [batch_size, shrinked_timesetps(here no shirnk), conv1_out_channels(hidRNN)]

      r = torch.cat((num_x,cat_x_embbed),2)
      out, _ = self.lstm(r)
      r = out[:, -1, :]
      r = self.dropout_layer(r)

      # no skip as no pattern of time-period
      
      # attention
      if (self.attn == 'multihead'):
        context_vector, _ = self.multihead(out, out, out)
        context_vector = context_vector.permute[:, -1, :]
        r = torch.cat((context_vector,r),1)
      res = self.fc(r);
      
      #highway auto-regressive unit
      if (self.highway_window > 0):
        # take only cltv_30d to do AR
        z = x_cltv        
        z = z.contiguous().view(-1, self.highway_window);
        z = self.highway(z);
        z = z.view(-1,self.output_dim);
        res = res + z;

      return res;

class LSTM_CNN_embedding(nn.Module):
  def __init__(self, seq_length, num_numcols, embedding_sizes, x_categoricals, categorical_groups,dropout, output_dim, hidden_dim, num_layers, threshold, flap_coefficient):
    super(LSTM_CNN_embedding, self).__init__()
    self.seq_length = seq_length
    self.num_numcols = num_numcols
    self.embedding_sizes = embedding_sizes
    self.x_categoricals = x_categoricals
    self.dropout = dropout
    self.output_dim = output_dim
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.threshold = threshold
    self.flap_coefficient = flap_coefficient    
    self.categorical_groups = categorical_groups
    self.device = torch.device('cuda')

    self.create_embedding = MultiEmbedding(embedding_sizes = self.embedding_sizes, x_categoricals = self.x_categoricals, categorical_groups = self.categorical_groups, embedding_paddings = [])
    self.create_embedding.eval().cuda()

    self.emb_out_dim = self.create_embedding.output_size

    self.lstnet = LSTM_CNN(
              seq_length = self.seq_length, 
              num_dim = self.num_numcols,
              emb_out_dim = self.emb_out_dim,
              output_dim = self.output_dim,
              hidRNN = 12,
              hidCNN = self.hidden_dim, # shirnk num features to dim of hidCNN, 23+13 = 36
              cnn_kernel_size = 1, # no shirnked timestep
              attn = None,
              highway_window = 0,
              dropout = self.dropout)
    self.lstnet.eval().cuda()
    
  def forward(self, cat_x, num_x):
    # creating embedding
    cat_x_embbed =  self.create_embedding.forward(cat_x.long()).to(device) 
    num_x.to(device)
    out = self.lstnet.forward(cat_x_embbed,num_x)
    
    # flapped if output < self.threshold (negative value), do this due to symmetry of MSE loss (tweedie not)
    out = torch.where(out < torch.Tensor([self.threshold]).to(device), torch.Tensor([self.flap_coefficient]).to(device)*out, out)   

    return out