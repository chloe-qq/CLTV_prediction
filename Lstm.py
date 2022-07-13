import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,TensorDataset
from torchinfo import summary


from Embedding import MultiEmbedding

class LSTM_embedding(nn.Module):
  def __init__(self, num_numcols, embedding_sizes, x_categoricals, dropout, output_dim, hidden_dim, num_layers, threshold, flap_coefficient):
    super(LSTM_embedding, self).__init__()
    self.num_numcols = num_numcols
    self.embedding_sizes = embedding_sizes
    self.x_categoricals = x_categoricals
    self.dropout = dropout
    self.output_dim = output_dim
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.threshold = threshold
    self.flap_coefficient = flap_coefficient
    
    self.categorical_groups = {}
    self.embedding_paddings = []
    self.concat_output = True
    self.device = torch.device('cuda')
    self.output_dim = output_dim
     
    self.create_embedding = MultiEmbedding(embedding_sizes = self.embedding_sizes, x_categoricals = self.x_categoricals, categorical_groups = {}, embedding_paddings = [])
    self.create_embedding.eval().cuda()
    self.input_dim = self.create_embedding.output_size + self.num_numcols

    self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, dropout=self.dropout, device = self.device)

    # Fully connected layer
    self.fc = nn.Linear(self.hidden_dim, self.output_dim, device = self.device)
    
  def forward(self, cat_x, num_x):
      # creating embedding
    cat_x_embbed =  self.create_embedding.forward(cat_x.long())
    x = torch.cat((num_x,cat_x_embbed),2).to(self.device)

    assert self.input_dim != 0,'input dim cannot be 0!'
    
    
    # Initializing hidden state for first input with zeros

    # lstm hidden unit
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
    c0 =  torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)

    # Forward propagation
    out, _ = self.lstm(x, (h0.detach(), c0.detach()))

    # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size) to feed into the fully connect layer
    out = out[:, -1, :]

    # Convert the final state to our desired output shape (batch_size, output_dim)
    out = self.fc(out)

    # due to the symmetry of MSE loss, if out < -0.01(threshold), set it to positive value
    out = torch.where(out < torch.Tensor([self.threshold]).to(device), torch.Tensor([self.flap_coefficient]).to(device)*out, out)

    return out

