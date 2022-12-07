# context-aware-ensemble
Source code for our paper [Context-Aware Ensemble Learning for Time Series](https://arxiv.org/abs/2211.16884), submitted to the journal IEEE Transactions on Neural Networks and Learning Systems.

### Sample Usage
(Additional comments and instructions can be found in the corresponding model)

For any weight constrained model from the LightGBM Ensemble:

```python
from lightgbm_ensemble import lgbm_ensemble_convex

parameters = {
    "predictor_count": 2,
    "learning_rate": 0.05,
    "max_depth": 3,
    "num_leaves": 20,
    "n_estimators": 100,
    "min_data_per_leaf": 10,
}

model = lgbm_ensemble_convex(**parameters)
model.fit(X_train, y_train)

train_preds = model.predict(X_train)
test_preds = model.predict(X_test)
```

For any weight constrained model from the MLP Ensemble:

```python
from mlp_ensemble import MLP_Ensembler

ens = MLP_Ensembler(num_features=(X_train.iloc[:,:-2].shape[1]), # Last two columns are the base model predictions
                    predictions=preds_list_1, # Predictions give in [preds_1, preds_2] format wherer preds_1 and preds_2 are 1-D numpy arrays
                    constraint="convex",
                    hidden_layer_sizes=(param_set[1],)
                    hidden_activations=nn.ReLU,
                    num_epochs=500,
                    learning_rate=0.05,
                    batch_size=128,
                    lambda_1=0.01,
                    lambda_2=0.05)

ens.fit(np.array(X_train.iloc[:,:-2]), np.array(y_train_1), verbose=False)
preds, p_weights = ens.predict(np.array(X_train.iloc[:,:-2]), np.array(X_train.iloc[:,-2:]))
fores, f_weights = ens.predict(np.array(X_test.iloc[:,:-2]), np.array(X_test.iloc[:,-2:]))
```
