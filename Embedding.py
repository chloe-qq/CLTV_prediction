import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,TensorDataset
from torchinfo import summary


# ref https://github.com/jdb78/pytorch-forecasting/blob/0dd04d3c310eb0e8e66e87c23165cbae1112f6c6/pytorch_forecasting/models/nn/embeddings.py#L9


class TimeDistributedEmbeddingBag(nn.EmbeddingBag):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if len(x.size()) <= 2:
            return super().forward(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        y = super().forward(x_reshape)
        y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)

        return y

class MultiEmbedding(nn.Module):

    concat_output: bool

    def __init__(
      self,
      embedding_sizes,
      x_categoricals: List[str] = None,
      categorical_groups: Dict[str, List[str]] = {},
      embedding_paddings: List[str] = [],
  ):
        """Embedding layer for categorical variables including groups of categorical variables. Enabled for static and dynamic categories (i.e. 3 dimensions for batch x time x categories).
        Args:
            embedding_sizes: dictionary of embedding sizes, e.g. ``{'cat1': (10, 3)}``
                  indicates that the first categorical variable has 10 unique values which are mapped to 3 embedding dimensions. 
                
            x_categoricals (List[str]): list of categorical variables
            
            categorical_groups (Dict[str, List[str]]): Defaults to empty dictionary.
              dictionary of categories that should be summed up in an embedding bag, 
              e.g. ``{'cat1': ['cat2', 'cat3']}`` indicates that a new categorical variable ``'cat1'`` is mapped to an embedding bag containing the second and third categorical variables.
              
            embedding_paddings (List[str]): list of categorical variables for which the value 0 is mapped to a zero embedding vector. Defaults to empty list.
        """
        super().__init__()

        self.embedding_sizes = embedding_sizes
        self.x_categoricals = x_categoricals
        self.categorical_groups = categorical_groups
        self.embedding_paddings = embedding_paddings
        self.concat_output = True
        self.device = torch.device('cuda')
        
        self.init_embeddings()

    def init_embeddings(self):
      self.embeddings = nn.ModuleDict()
      for name in self.embedding_sizes.keys():
        embedding_size = self.embedding_sizes[name][0]
        embedding_dim = self.embedding_sizes[name][1]

        if name in self.categorical_groups:  # embedding bag if related embeddings
          self.embeddings[name] = TimeDistributedEmbeddingBag(
              self.embedding_sizes[name][0], embedding_size, mode="sum", batch_first=True
          )
        else:
          if name in self.embedding_paddings:
              padding_idx = 0
          else:
              padding_idx = None
          self.embeddings[name] = nn.Embedding(
            embedding_size,
            embedding_dim,
            max_norm=math.sqrt(embedding_dim),
            padding_idx=padding_idx,
            device=device
          )

    def names(self):
      return list(self.keys())

    def items(self):
      return self.embeddings.items()

    def keys(self):
      return self.embeddings.keys()

    def values(self):
      return self.embeddings.values()

    def __getitem__(self, name: str):
      return self.embeddings[name]

    @property
    def input_size(self) -> int:
      return len(self.x_categoricals)

    @property
    def output_size(self):
      output_dim = 0
      if self.concat_output:
        for s in self.embedding_sizes.values():
          output_dim += s[1]
        return output_dim
      else:
        return {name: s[1] for name, s in self.embedding_sizes.items()}

    def forward(self, x: torch.Tensor):
      """
      Args:
        x (torch.Tensor): input tensor of shape batch x (optional) time x categoricals in the order of ``x_categoricals``.
      Returns:
        the embedding of shape batch x (optional) time x sum(embedding_sizes).
        Query attribute ``output_size`` to get the size of the output(s).
      """
      input_vectors = {}
      for name, emb in self.embeddings.items():
        if name in self.categorical_groups:
          input_vectors[name] = emb(
              x[
                  ...,
                  [self.x_categoricals.index(cat_name) for cat_name in self.categorical_groups[name]],
              ]
          )
        else:
          # add relu to embedding layer
          input_vectors[name] = F.mish(emb(x[..., self.x_categoricals.index(name)]))
          #input_vectors[name] = emb(x[..., self.x_categoricals.index(name)])

      if self.concat_output:  # concatenate output
        return torch.cat(list(input_vectors.values()), dim=-1)
      else:
        return input_vectors