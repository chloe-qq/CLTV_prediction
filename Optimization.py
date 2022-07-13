import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,TensorDataset
from torchinfo import summary


class Optimization:
  def __init__(self, model, loss_fn, optimizer, patience):
    self.model = model
    self.loss_fn = loss_fn
    self.optimizer = optimizer
    self.patience = patience
    self.train_losses = []
    self.val_losses = []

  def train_step(self, cat_x, num_x, y):
  # Sets model to train mode
    self.model.train()

    # Makes predictions
    yhat = self.model(cat_x, num_x)

    # Computes loss
    loss = self.loss_fn(y, yhat)
    
    # Computes gradients
    loss.backward()

    # Updates parameters and zeroes gradients
    self.optimizer.step()
    self.optimizer.zero_grad()

    # Returns the loss
    return loss.item()

  def train(self, train_loader, val_loader, batch_size=64, n_epochs=50, n_features=1):
    # apply early stopping
    trigger_time = 0
    
    for epoch in range(1, n_epochs + 1):
      batch_losses = []
      for cat_x_batch, num_x_batch, y_batch in train_loader:
        cat_x_batch = cat_x_batch.to(device)
        num_x_batch = num_x_batch.to(device)
        y_batch = y_batch.to(device)
        loss = self.train_step(cat_x_batch, num_x_batch, y_batch)
        batch_losses.append(loss)
        
      training_loss = np.mean(batch_losses)
      self.train_losses.append(training_loss)

      with torch.no_grad():
        batch_val_losses = []
        for cat_x_val, num_x_val, y_val in val_loader:
          cat_x_val = cat_x_val.to(device)
          num_x_val = num_x_val.to(device)
          y_val = y_val.to(device)
          self.model.eval()
          yhat = self.model(cat_x_val,num_x_val)
          val_loss = self.loss_fn(y_val, yhat).item()
          batch_val_losses.append(val_loss)
        validation_loss = np.mean(batch_val_losses)
        
        if (epoch > 1 and validation_loss > self.val_losses[-1]):
          trigger_times += 1
          if (trigger_times >= self.patience):
            print('Early stopping!')
            break
        else:
          trigger_times = 0
        self.val_losses.append(validation_loss)

      print(f"[{epoch}/{n_epochs}] Training loss: {training_loss:.8f}\t Validation loss: {validation_loss:.8f}")

    #torch.save(self.model.state_dict(), model_path)
    
    
  def evaluate(self, test_loader, batch_size=1, n_features=1):
    with torch.no_grad():
      
      predictions = []
      values = []
      for cat_x_test, num_x_test, y_test in test_loader:
          cat_x_test = cat_x_test.to(device)
          num_x_test = num_x_test.to(device)
          y_test = y_test.to(device)
          self.model.eval()
          yhat = self.model(cat_x_test,num_x_test)
          # if yhat < 0, set it to 0 
          yhat = torch.where(yhat < torch.Tensor([0.0]).to(device), torch.Tensor([0.0]).to(device), yhat)

          predictions.append(yhat.cpu().detach().numpy())
          values.append(y_test.cpu().detach().numpy())

    return predictions, values
  
  
  def plot_losses(self):
    plt.plot(self.train_losses, label="Training loss")
    print(f'train loss: {self.train_losses}')
    plt.plot(self.val_losses, label="Validation loss")
    print(f'validation loss: {self.val_losses}')
    plt.legend()
    plt.title("Losses")
    plt.show()
    
    #plt.close()