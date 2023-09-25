# context-time-aware-ensemble
Source code for our paper Time-Aware and Context-Sensitive Ensemble Learning for Sequential Data, published by the journal IEEE Transactions on Artificial Intelligence.
[PAPER URL](https://doi.org/10.1109/TAI.2023.3319308)

### Sample Usage
(Additional comments and instructions can be found in the corresponding model code)

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

# First "predictor_count" columns of X_train and X_test are the base model predictions

model = lgbm_ensemble_convex(**parameters)
model.fit(X_train, y_train)

train_preds = model.predict(X_train)
test_preds = model.predict(X_test)
```

For any weight constrained model from the MLP Ensemble:

```python
from mlp_ensemble import MLP_Ensembler

# Last "predictor_count" columns of X_train and X_test are the base model predictions
# "predictions" is a list, in the format of [preds_1, preds_2, ..] where preds_{i} are 1-D numpy arrays corresponding to base model predictions

ens = MLP_Ensembler(num_features=(X_train.iloc[:,:-2].shape[1]),
                    predictions=predictions,
                    constraint="convex",
                    hidden_layer_sizes=(16,)
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
