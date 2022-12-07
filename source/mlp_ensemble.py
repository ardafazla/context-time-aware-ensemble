import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error as mse


class MLP_Ensembler(nn.Module):
    def __init__(self, num_features, predictions, constraint="unconstrained",
                 hidden_layer_sizes=(64, 16), hidden_activations=nn.ReLU,
                 num_epochs=10_000, learning_rate=3e-3, batch_size=128,
                 lambda_1=1e-2, lambda_2=1e-2, loss_criterion=nn.MSELoss,
                 optimizer=optim.Adam):
        """
        MLP Ensembler: given `predictions` of base regressors, trained to find
        a mapping for a good set of weights to combine them. Various
        constraints can be imposed on the blending vector.
        
        Parameters
        ----------
        num_features : int
            Number of features the model will be trained with.
        
        predictions : list of pd.Series or list of np.ndarray
            Base regressors' predictions for the training data.
        
        constraint : str, default="unconstrained"
            One of "unconstrained", "affine", "convex". If "unconstrained",
            last layer of the network is activated with identity. If "affine",
            still identity; but the weight vector is divided by its sum so it
            sums to 1 prior to blending. If "convex", softmax activation is
            used, i.e., weights sum up to 1 *and* they are nonnegative.
        
        hidden_layer_sizes : tuple of int, default=(64, 16)
            Determines both the number of hidden layers and the number of units
            per layer. Excludes the input and output layers.
        
        hidden_activations : torch.nn.<ACT> or list of torch.nn.<ACT>,
                             default=torch.nn.ReLU
            Nonlinearities used in the hidden layers. Should be given
            uninitialized. If a list, length must match that of
            `hidden_layer_sizes`; otherwise, repeated for each layer.
        
        num_epochs : int, default=10_000
            Number of epochs to train.
        
        learning_rate : float, default=3e-3
            Passed to `optimizer`.
        
        batch_size : int, default=128
            Number of samples per batch.
        
        lambda_1 : float, default=1e-2
            L1 regularization parameter.
            
        lambda_2 : float, default=1e-2
            L2 regularization parameter.
        
        loss_criterion : torch.nn.<CRI>, default=torch.nn.MSELoss
            Loss function. Should be uninitialized.
        
        optimizer : torch.optim.<OPT>, default=torch.optim.Adam
            Optimizer.
        
        Notes
        -----
        - Number of inputs are determined from `num_features`
        - Number of outputs are determined from `len(predictions)`
        - Network's output are weights and `predictions` are weighted with 
        these and then fed to the `loss_criterion` agains the ground truths.
        """
        super().__init__()
        
        self.constraint = constraint
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_activations = hidden_activations
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        
        # If a single nonlinearity is given, repeat it for each hidden layer
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations
                                  for _ in enumerate(hidden_layer_sizes)]
        
        # Start forming network layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.BatchNorm1d(num_features))
        
        # Add linear layers along with activation functions
        prev = num_features
        for hidden_size, hidden_act in zip(hidden_layer_sizes, hidden_activations):
            self.layers.append(nn.Linear(prev, hidden_size))
            self.layers.append(hidden_act())
            prev = hidden_size
        
        # Output layer
        self.layers.append(nn.Linear(prev, len(predictions)))
        
        if self.constraint == "convex":
            self.layers.append(nn.Softmax(dim=1))

        # Register to `self` for further use
        self.loss_function = loss_criterion()
        self.optimizer = optimizer(self.parameters(),
                                   lr=learning_rate,
                                   weight_decay=lambda_2)
        
        # Stack `predictions` to `(n_samples, len(predictions))`
        # `np.asarray` is due to github.com/pytorch/pytorch/issues/51112
        self.predictions = torch.hstack(
                                [torch.as_tensor(np.asarray(pred), dtype=torch.double).view(-1, 1)
                                 for pred in predictions])

    def forward(self, x):
        # forward pass
        for layer in self.layers:
            x = layer(x)
        
        # in case "affine", divide by the sum
        if self.constraint == "affine":
            temp = torch.sum(x, axis=1)
            x[:,0] /= temp
            x[:,1] /= temp
            # x /= x.sum(1)[:, None]
        return x
  
    def get_data_loader(self, X, y):
        """
        Prepares a `DataLoader` given 2D X and 1D y.
        """
        X = torch.as_tensor(np.asarray(X), dtype=torch.double)
        X = torch.hstack((X, self.predictions))
        y = torch.as_tensor(np.asarray(y), dtype=torch.double)
        ds = torch.utils.data.TensorDataset(X, y)
        dl = torch.utils.data.DataLoader(ds,
                                         batch_size=self.batch_size,
                                         shuffle=True)
        return dl

    def get_loss(self, truths, preds, weights):
        """
        Returns the loss computed with `self.loss_function` optionally added
        the L1 regularization with penalty multiplier `self.lambda_1`.
        """
        
        # base loss
        weighted_preds = (weights * preds).sum(dim=1)
        loss = self.loss_function(weighted_preds, truths)
        
        # L1 Regularization?
        if self.lambda_1 > 0:
            # get L1 over weights
            l1_reg = torch.tensor(0., requires_grad=True)
            for name, param in self.named_parameters():
                if "weight" in name:
                    l1_reTrueg = l1_reg + torch.norm(param, p=1)
            # return regularized loss (L2 is applied with optimizer)
            return loss + self.lambda_1 * l1_reg
        else:
            return loss

    def fit(self, X, y, verbose=True):
        """
        Training happens here
        """
        # train mode on
        self.train()
        self.double()
        
        # prepare the data loader
        train_loader = self.get_data_loader(X, y)
        num_models = self.predictions.shape[1]
        
        for epoch_idx in range(self.num_epochs):
            epoch_loss = 0.
            for batch_idx, (xb, yb) in enumerate(train_loader):
                
                # Forward pass and get the loss
                xb = xb.double()
                xb, pb = xb[:, :-num_models], xb[:, -num_models:]
                wb = self(xb)
                
                loss = self.get_loss(truths=yb, preds=pb, weights=wb)           

                # Backward pass
                self.optimizer.zero_grad() 
                loss.backward()                            
                self.optimizer.step()                      

                # save the loss to report later
                epoch_loss += loss.item()
            
            if verbose:
                print(f"Epoch idx: {epoch_idx+1}, Loss: {epoch_loss/len(yb):.3f}", 
                      end="\r", flush=True)

    def predict(self, X, base_preds, _round=True):
        """
        Return the predictions for `X` as well as the weights used.
        """
        if isinstance(base_preds, (list, tuple)):
            base_preds = torch.hstack([torch.as_tensor(np.asarray(pred)).view(-1, 1)
                                       for pred in base_preds])
        else:
            base_preds = torch.as_tensor(base_preds)

        self.eval()
        with torch.no_grad():    
            weights = self(torch.as_tensor(np.asarray(X)))
            if _round:
                weights = torch.round(weights, decimals=3)
            preds = (weights * base_preds).sum(dim=1)
        return preds.numpy(), weights.numpy()
    
    def score(self, X, y, base_preds, loss_fun=mse):
        """
        Predict the samples in `X` and weight the `base_preds` accordginly.
        Then compare it with ground truths `y` using supplied loss function,
        and return it.
        """
        preds, _ = self.predict(X, base_preds)
        return loss_fun(y, preds)
    