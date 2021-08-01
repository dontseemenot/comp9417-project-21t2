
# %% Model creation 
# Put classifiers and search parameters in here
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchsummary
from torch.utils.data import TensorDataset, DataLoader

from sklearn.utils.estimator_checks import check_estimator

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, y_pred, y_true):
        L = self.loss(y_pred, y_true)
        return L
    
    def forward_no_seq(self, y_pred, y_true):
        return self.loss(y_pred, y_true)

def get_accuracy(y_pred, y_true):
    y_pred_cls = y_pred.argmax(dim=-1)
    acc = (y_pred_cls == y_true).float()
    return acc.mean()

class Model(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_hidden, hidden_activ):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.act1 = hidden_activ()
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.act2 = hidden_activ()
        self.fc3 = nn.Linear(n_hidden, n_outputs)
        self.act3 = hidden_activ()
        # self.act2 = nn.Sigmoid()
    
    # Takes in an input of size (B,F)
    # B = batch size
    # F = total features
    def forward(self, x):
        x0 = self.fc1(x)
        x0 = self.act1(x0)

        x1 = self.fc2(x0)
        x1 = self.act2(x1)

        x2 = self.fc3(x1)
        x2 = self.act3(x2)

        y = x2
        return y

## Model training and testing
def train(model, optimiser, loss_func, train_dl, total_epochs):
    model.train()
    for epoch in range(total_epochs):
        metrics = []
        for x, y_true in train_dl:
            optimiser.zero_grad()
            y_pred = model.forward(x)
            loss = loss_func.forward(y_pred, y_true)
            #loss = loss_func.forward_no_seq(y_pred, y_true)
            loss.backward()
            optimiser.step()

            acc = get_accuracy(y_pred, y_true)

            metrics.append(np.array([loss.detach().numpy(), acc.detach().numpy()]))
            print(f"\repoch={epoch:3d}, loss={loss:.4e}, acc={acc:.3f}", end="")
            
        metrics = np.array(metrics)
        train_loss, train_acc = np.mean(metrics, axis=0).flatten()
        # test_loss, test_acc = test(model, loss_func, test_dl)
        print(f"\repoch={epoch:3d}, "+\
              f"loss={train_loss:.4e}, acc={train_acc:.3f}", end="")

    print(f"\repoch={epoch:3d}, "+\
            f"loss={train_loss:.4e}, acc={train_acc:.3f}")
            

def test(model, loss_func, test_dl):
    model.eval()
    metrics = []
    with torch.no_grad():
        for x, y_true in test_dl:
            y_pred = model.forward(x)
            loss = loss_func.forward(y_pred, y_true)
            acc = get_accuracy(y_pred, y_true)

            metrics.append(np.array([loss.detach().numpy(), acc.detach().numpy()]))
            #print(f"\repoch={epoch:3d}, loss={loss:.4e}, acc={acc:.3f}", end="")
    return np.mean(np.array(metrics), axis=0).flatten()

class PytorchEstimator:
    def __init__(self, hidden_count=100, activation=nn.ReLU, 
                       learning_rate=0.001, batch_size=64, total_epochs=200):
        self.hidden_count = hidden_count
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.total_epochs = total_epochs
    
    def get_params(self, deep=True):
        return {
            'hidden_count': self.hidden_count, 
            'activation': self.activation,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'total_epochs': self.total_epochs}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def fit(self, X, Y):
        if np.iscomplex(X).any() or np.iscomplex(Y).any():
            raise ValueError("Complex data not supported")

        # create dataset as a dataloader
        x, y = torch.tensor(X.astype(np.float32)), torch.tensor(Y.astype(np.int64))
        train_ds = TensorDataset(x, y)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        nb_features = X.shape[-1]

        model = Model(
            n_inputs=nb_features, n_outputs=5, 
            n_hidden=self.hidden_count, hidden_activ=self.activation)

        optimiser = optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_func = Loss()

        train(model, optimiser, loss_func, train_dl, self.total_epochs)
        model.eval()
        self.model = model
        return self
    
    def predict(self, X):
        if np.iscomplex(X).any():
            raise ValueError("Complex data not supported")
            
        x = torch.Tensor(X.astype(np.float32))
        self.model.eval()
        y = self.model.forward(x)
        y = y.detach().numpy()
        return np.argmax(y, axis=-1)
    
    def predict_proba(self, X):
        if np.iscomplex(X).any():
            raise ValueError("Complex data not supported")

        x = torch.Tensor(X.astype(np.float32))
        self.model.eval()
        y = self.model.forward(x)
        y = torch.log_softmax(y, dim=-1)
        y = y.detach().numpy()
        return y

