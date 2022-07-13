class LSTM_ATTENTION_embedding(nn.Module):
  def __init__(self, num_numcols, embedding_sizes, x_categoricals, dropout, output_dim, hidden_dim, num_layers, threshold, flap_coefficient, attn):
    super(LSTM_ATTENTION_embedding, self).__init__()
    self.num_numcols = num_numcols
    self.embedding_sizes = embedding_sizes
    self.x_categoricals = x_categoricals
    self.dropout = dropout
    self.output_dim = output_dim
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.threshold = threshold
    self.flap_coefficient = flap_coefficient
    self.attn = attn
    
    self.categorical_groups = {}
    self.embedding_paddings = []
    self.concat_output = True
    self.device = torch.device('cuda')
    self.output_dim = output_dim
     
    self.create_embedding = MultiEmbedding(embedding_sizes = self.embedding_sizes, x_categoricals = self.x_categoricals, categorical_groups = {}, embedding_paddings = [])
    self.create_embedding.eval().cuda()

    self.input_dim = self.create_embedding.output_size + self.num_numcols
    
    self.dropout_layer = nn.Dropout(p = self.dropout)
    self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, dropout=self.dropout, device = self.device)
    
    self.multihead = None
    if (self.attn):
      self.multihead = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=3, device = self.device, batch_first = True)
      self.fc = nn.Linear(in_features=self.hidden_dim*2, out_features=self.output_dim, device = self.device)
    else:
      self.fc = nn.Linear(self.hidden_dim, self.output_dim, device = self.device)
    
  def forward(self, cat_x, num_x):
      # creating embedding
    cat_x_embbed =  self.create_embedding.forward(cat_x.long())
    x = torch.cat((num_x,cat_x_embbed),2).to(self.device)

    assert self.input_dim != 0,'input dim cannot be 0!'
    
    
    # Initializing hidden state for first input with zeros: lstm hidden unit
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
    c0 =  torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)

    # Forward propagation
    output, _ = self.lstm(x, (h0.detach(), c0.detach()))
    
    # attention
    if (self.attn == 'scaled_dot'):
      # input from the previous state + dynamic context vector
      # (batch_size, time_steps, hidden_size)
      score_first_part = output
      # (batch_size, hidden_size)
      h_t = output[:,-1,:]
      # (batch_size, time_steps)
      score = torch.bmm(score_first_part, h_t.unsqueeze(2)).squeeze(2)/ np.sqrt(self.hidden_dim)
      attention_weights = F.softmax(score, dim=1)
      # (batch_size, hidden_size)
      context_vector = torch.bmm(output.permute(0,2,1), attention_weights.unsqueeze(2)).squeeze(2)
      # (batch_size, hidden_size*2)
      pre_activation = torch.cat((context_vector, h_t), dim=1)

    if (self.attn == 'multihead'):
      h_t = output[:,-1,:]

      context_vector, _ = self.multihead(output, output, output)
      # Reshaping the outputs in the shape of (batch_size,  hidden_size)
      context_vector = context_vector[:, -1, :]
      pre_activation = torch.cat((context_vector,h_t),dim=1)

    # Convert the final state to our desired output shape (batch_size, output_dim)
    if (self.attn):
      assert pre_activation is not None, 'pre_activation not calculated!'
      out = self.fc(pre_activation)

    else:
      # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size) to feed into the fully connect layer
      out = output[:, -1, :]
      out = self.fc(out)

    # due to the symmetry of MSE loss, if out < -0.5(threshold), set it to positive value * 0.8
    out = torch.where(out < torch.Tensor([self.threshold]).to(device), torch.Tensor([self.flap_coefficient]).to(device)*out, out)

    return out