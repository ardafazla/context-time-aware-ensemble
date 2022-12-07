import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None

import lightgbm as lgb
from scipy.special import softmax



class lgbm_ens_convex():
    """
    Custom implementation of LightGBM Ensemble Convex
    Weight condition: Convex, meaning all weights are between [0,1] and sum up to 1
    """

    def __init__(
        self,
        predictor_count=2,
        learning_rate: float = 0.001,
        max_depth: int = 5,
        num_leaves: int = 15,
        n_estimators: int = 100,
        min_data_per_leaf: int = 10,
        verbose: int = -1,  # 0
    ):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.n_estimators = n_estimators
        self.min_data_per_leaf = min_data_per_leaf
        self.verbose = verbose
        self.predictor_count = predictor_count

        self._is_fitted = False
        self.train_weights = None
        self.predict_weights = None

    def fit(self, X, y):

        self.model = None

        # Training Predictions of Base Algorithms example format:
        # [preds_lightgbm, preds_sarimax]
        eval_preds = []

        for i in range(self.predictor_count):
            eval_preds.append(X[:, -i - 1])
            
        eval_preds = list(reversed(eval_preds))
        
        errors = pd.DataFrame()

        for i in range(self.predictor_count):
            errors[str(i)] = y - eval_preds[i]

        lgb_train = lgb.Dataset(pd.DataFrame(X[:, : -self.predictor_count]))
        errors = errors.to_numpy()

        def my_softmax(y_hat, dtrain):
            n_data = len(X)
            n_classes = len(eval_preds)
            
            y_hat_nexp = np.transpose(y_hat.reshape((n_classes, n_data)))
            y_hat = np.exp(y_hat_nexp)
            probs = y_hat / np.sum(y_hat, axis=1).reshape([-1, 1])
            rowsumerrors = np.tile(
                (probs * errors).sum(axis=1).reshape(n_data, 1), (1, n_classes)
            )
            
            grad = 2 * rowsumerrors * probs * (errors - rowsumerrors)
            hess = grad * (1 - 2 * probs) + 2 * ( probs * (errors - rowsumerrors) ) ** 2

            return np.transpose(grad).flatten(), np.transpose(hess).flatten()

        clf_params = {
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "n_estimators": self.n_estimators,
            "min_data_per_leaf": self.min_data_per_leaf,
            "num_class": self.predictor_count,
            "verbose": self.verbose,
        }

        self.model = lgb.train(
            params=clf_params,
            train_set=lgb_train,
            valid_sets=[lgb_train],
            fobj=my_softmax,
        )
        
        self._is_fitted = True

        # Weights
        self.train_weights = softmax(self.model.predict(X[:, : -self.predictor_count]), axis=1)
        self.predict_weights = None

        return self

    def predict(self, X):

        if not self._is_fitted:
            raise ValueError("Model was not fit.")

        # Test Predictions of Base Algorithms example format:
        # [fors_lightgbm, fors_sarimax]
        self.predict_weights = softmax(self.model.predict(X[:, : -self.predictor_count]), axis=1)
        
        y_pred = np.sum(
            X[:, -self.predictor_count :] * self.predict_weights,
            axis=1,
        )

        return y_pred

    def plot_train_weights(self):
        
        fig, ax = plt.subplots(self.predictor_count, 1, figsize=(20, 10))
        
        for i in range(self.predictor_count):
            ax[i].plot(np.float16(self.train_weights[:, i]))
            ax[i].set_title(f"LGBM-Ensemble weight_{i}")
            ax[i].grid(True)
            ax[i].legend()
            
        plt.tight_layout()
        
    def plot_test_weights(self):
        
        fig, ax = plt.subplots(self.predictor_count, 1, figsize=(20, 10))
        
        for i in range(self.predictor_count):
            ax[i].plot(np.float16(self.predict_weights[:, i]))
            ax[i].set_title(f"LGBM-Ensemble weight_{i}")
            ax[i].grid(True)
            ax[i].legend()
            
        plt.tight_layout()
        


class lgbm_ens_affine():
    """
    Custom implementation of LightGBMEnsemble Affine
    Weight condition: Affine, meaning all weights sum up to 1
    """

    def __init__(
        self,
        predictor_count=2,
        learning_rate: float = 0.001,
        max_depth: int = 5,
        num_leaves: int = 15,
        n_estimators: int = 100,
        min_data_per_leaf: int = 10,
        verbose: int = -1,  # 0
    ):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.n_estimators = n_estimators
        self.min_data_per_leaf = min_data_per_leaf
        self.verbose = verbose
        self.predictor_count = predictor_count
        
        self.start = True
        self._is_fitted = False
        self.train_weights = None
        self.predict_weights = None

    def affine(self, y_hat, eps=0.0):
        return y_hat / (y_hat.sum(axis=1) + eps)[:, np.newaxis]

    def fit(self, X, y):

        self.model = None

        # Training Predictions of Base Algorithms example format:
        # [preds_lightgbm, preds_sarimax]
        eval_preds = []

        for i in range(self.predictor_count):
            eval_preds.append(X[:, -i - 1])
            
        eval_preds = list(reversed(eval_preds))
        
        errors = pd.DataFrame()

        for i in range(self.predictor_count):
            errors[str(i)] = y - eval_preds[i]

        lgb_train = lgb.Dataset(pd.DataFrame(X[:, : -self.predictor_count]))
        errors = errors.to_numpy()

        def my_softmax(y_hat, dtrain):
            n_data = len(X)
            n_classes = len(eval_preds)

            if self.start:
                y_hat += 0.5
                self.start = False
            
            y_hat = np.transpose(y_hat.reshape((n_classes, n_data)))
            m = np.sum(y_hat, axis=1)
            
            m_tile = np.tile((1/m).reshape(-1,1), (1,n_classes))
            
            probs = y_hat / m.reshape([-1, 1])

            preds = np.zeros((n_classes, n_data))

            for i in range(n_classes):
                preds[i, :] = eval_preds[i]

            preds = np.transpose(preds) # n_classes x n_data
            
            total_preds = np.sum(preds * probs, axis=1)
            total_preds_tile = np.tile(total_preds.reshape(-1,1), (1, n_classes))

            total_errors_tile = np.tile(y.reshape(-1,1), (1, n_classes)) - total_preds_tile

            grad = 2 * total_errors_tile * m_tile * (total_preds_tile - preds)
            hess = 2 * (m_tile ** 2) * (total_preds_tile - preds) * [ (total_preds_tile - preds) - 2 * total_errors_tile]
            
            return np.transpose(grad).flatten(), np.transpose(hess).flatten()

        clf_params = {
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "n_estimators": self.n_estimators,
            "min_data_per_leaf": self.min_data_per_leaf,
            "num_class": self.predictor_count,
            "verbose": self.verbose,
        }

        self.model = lgb.train(
            params=clf_params,
            train_set=lgb_train,
            valid_sets=[lgb_train],
            fobj=my_softmax,
        )
        
        self._is_fitted = True

        # Weights
        self.train_weights = self.affine(self.model.predict(X[:, : -self.predictor_count]))
        self.predict_weights = None

        return self

    def predict(self, X):

        if not self._is_fitted:
            raise ValueError("Model was not fit.")

        # Test Predictions of Base Algorithms example format:
        # [fors_lightgbm, fors_sarimax]
        self.predict_weights = self.affine(self.model.predict(X[:, : -self.predictor_count]))
        
        y_pred = np.sum(X[:, -self.predictor_count :] * self.predict_weights, axis=1)

        return y_pred
    
    def plot_train_weights(self):
        
        fig, ax = plt.subplots(self.predictor_count, 1, figsize=(20, 10))
        
        for i in range(self.predictor_count):
            ax[i].plot(np.float16(self.train_weights[:, i]))
            ax[i].set_title(f"LGBM-Ensemble weight_{i}")
            ax[i].grid(True)
            ax[i].legend()
            
        plt.tight_layout()
        
    def plot_test_weights(self):
        
        fig, ax = plt.subplots(self.predictor_count, 1, figsize=(20, 10))
        
        for i in range(self.predictor_count):
            ax[i].plot(np.float16(self.predict_weights[:, i]))
            ax[i].set_title(f"LGBM-Ensemble weight_{i}")
            ax[i].grid(True)
            ax[i].legend()
            
        plt.tight_layout()
        


class lgbm_ensemble_affine_project():
    """
    Custom implementation of LightGBMEnsemble Affine Projection
    Addition: another version of the affine constraint solution, 
    based on orthogonal projection of the unconstrained weights to affine subspace
    Weight condition: Affine, meaning all weights sum up to 1
    """
    
    def __init__(
        self,
        predictor_count=2,
        learning_rate: float = 0.001,
        max_depth: int = 5,
        num_leaves: int = 15,
        n_estimators: int = 100,
        min_data_per_leaf: int = 10,
        verbose: int = -1,  # 0
    ):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.n_estimators = n_estimators
        self.min_data_per_leaf = min_data_per_leaf
        self.verbose = verbose
        self.predictor_count = predictor_count

        self._is_fitted = False
        self.train_weights = None
        self.predict_weights = None
        
        
    def transform(self, y_hat):
        weights = np.zeros((y_hat.shape[0],y_hat.shape[1]))
        weights[:,0] = 0.5*y_hat[:,0] - 0.5*y_hat[:,1] + 0.5
        weights[:,1] = -0.5*y_hat[:,0] + 0.5*y_hat[:,1] + 0.5
        return weights


    def fit(self, X, y):

        self.model = None

        # Training Predictions of Base Algorithms example format:
        # [preds_lightgbm, preds_sarimax]
        eval_preds = []

        for i in range(self.predictor_count):
            eval_preds.append(X[:, -i - 1])
            
        eval_preds = list(reversed(eval_preds))
        
        errors = pd.DataFrame()

        for i in range(self.predictor_count):
            errors[str(i)] = y - eval_preds[i]

        lgb_train = lgb.Dataset(pd.DataFrame(X[:, :-self.predictor_count]))
        errors = errors.to_numpy()
        
        def my_softmax(y_hat, dtrain):
            n_data = len(X)
            self.n_data = n_data
            n_classes = len(eval_preds)

            preds = np.zeros((n_classes, n_data))

            for i in range(n_classes):
                preds[i, :] = eval_preds[i]

            preds = np.transpose(preds) # n_classes x n_data
            probs = np.zeros((n_data, n_classes))
            y_hat = np.transpose(y_hat.reshape((n_classes, n_data)))

            probs[:, 0] = 0.5 * y_hat[:, 0] - 0.5 * y_hat[:, 1] + 0.5
            probs[:, 1] = -0.5 * y_hat[:, 0] + 0.5 * y_hat[:, 1] + 0.5
            
            first_half = np.tile(
                y.reshape(n_data, 1) - ((probs * preds).sum(axis=1).reshape(n_data, 1)),
                (1, n_classes))

            second_half = np.zeros((n_data, n_classes))
            second_half[:, 0] = - preds[:, 0] + preds[:, 1]
            second_half[:, 1] = preds[:, 0] - preds[:, 1]

            grad = first_half * second_half
            hess = 0.5 * (second_half ** 2)

            return np.transpose(grad).flatten(), np.transpose(hess).flatten()

        clf_params = {
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "n_estimators": self.n_estimators,
            "min_data_per_leaf": self.min_data_per_leaf,
            "num_class": self.predictor_count,
            "verbose": self.verbose,
        }

        self.model = lgb.train(
            params=clf_params,
            train_set=lgb_train,
            fobj=my_softmax,
        )
        
        self._is_fitted = True

        # Weights
        self.train_weights = self.transform(self.model.predict(X[:, :-self.predictor_count]))
        self.predict_weights = None

        return self

    def predict(self, X):

        if not self._is_fitted:
            raise ValueError("Model was not fit.")

        # Test Predictions of Base Algorithms example format:
        # [fors_lightgbm, fors_sarimax]
        self.predict_weights = self.transform(self.model.predict(X[:, :-self.predictor_count]))
        
        y_pred = np.sum(
            X[:, -self.predictor_count:] * self.predict_weights,
            axis=1,
        )

        return y_pred
    
    def plot_train_weights(self):
        
        fig, ax = plt.subplots(self.predictor_count, 1, figsize=(20, 10))
        
        for i in range(self.predictor_count):
            ax[i].plot(np.float16(self.train_weights[:, i]))
            ax[i].set_title(f"LGBM-Ensemble weight_{i}")
            ax[i].grid(True)
            ax[i].legend()
            
        plt.tight_layout()
        
    def plot_test_weights(self):
        
        fig, ax = plt.subplots(self.predictor_count, 1, figsize=(20, 10))
        
        for i in range(self.predictor_count):
            ax[i].plot(np.float16(self.predict_weights[:, i]))
            ax[i].set_title(f"LGBM-Ensemble weight_{i}")
            ax[i].grid(True)
            ax[i].legend()
            
        plt.tight_layout()
        
        
        
# No Constraint without Base Preds
class lgbm_ens_nc():
    """
    Custom implementation of LightGBMEnsemble No Constraint
    Weight condition: No constraint on weights
    """

    def __init__(
        self,
        predictor_count=2,
        learning_rate: float = 0.001,
        max_depth: int = 5,
        num_leaves: int = 15,
        n_estimators: int = 100,
        min_data_per_leaf: int = 10,
        verbose: int = -1,  # 0
    ):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.n_estimators = n_estimators
        self.min_data_per_leaf = min_data_per_leaf
        self.verbose = verbose
        self.predictor_count = predictor_count

        self.start = True
        self._is_fitted = False
        self.train_weights = None
        self.predict_weights = None

    def fit(self, X, y):

        self.model = None

        # Training Predictions of Base Algorithms example format:
        # [preds_lightgbm, preds_sarimax]
        eval_preds = []

        for i in range(self.predictor_count):
            eval_preds.append(X[:, -i - 1])
            
        eval_preds = list(reversed(eval_preds))
        
        errors = pd.DataFrame()

        for i in range(self.predictor_count):
            errors[str(i)] = y - eval_preds[i]

        lgb_train = lgb.Dataset(pd.DataFrame(X[:, : -self.predictor_count]))
        errors = errors.to_numpy()

        def my_softmax(y_hat, dtrain):
            n_data = len(X)
            n_classes = len(eval_preds)
            
            if self.start:
                y_hat += 0.5
                self.start = False
            
            y_hat = np.transpose(y_hat.reshape((n_classes, n_data)))
            
            probs = y_hat
            
            preds = np.zeros((n_classes, n_data))

            for i in range(n_classes):
                preds[i, :] = eval_preds[i]

            preds = np.transpose(preds) # n_classes x n_data
            
            total_preds = np.sum(preds * probs, axis=1)
            total_preds_tile = np.tile(total_preds.reshape(-1,1), (1, n_classes))
            total_errors_tile = np.tile(y.reshape(-1,1), (1, n_classes)) - total_preds_tile

            grad = -2 * probs * total_errors_tile
            hess = 2 * (probs ** 2)
            
            return np.transpose(grad).flatten(), np.transpose(hess).flatten()

        clf_params = {
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "n_estimators": self.n_estimators,
            "min_data_per_leaf": self.min_data_per_leaf,
            "num_class": self.predictor_count,
            "verbose": self.verbose,
        }

        self.model = lgb.train(
            params=clf_params,
            train_set=lgb_train,
            valid_sets=[lgb_train],
            fobj=my_softmax,
        )
        
        self._is_fitted = True

        # Weights
        self.train_weights = self.model.predict(X[:, : -self.predictor_count])
        self.predict_weights = None

        return self

    def predict(self, X):

        if not self._is_fitted:
            raise ValueError("Model was not fit.")

        # Test Predictions of Base Algorithms example format:
        # [fors_lightgbm, fors_sarimax]
        self.predict_weights = self.model.predict(X[:, : -self.predictor_count])
        
        y_pred = np.sum(
            X[:, -self.predictor_count :] * self.predict_weights,
            axis=1,
        )

        return y_pred
    
    def plot_train_weights(self):
        
        fig, ax = plt.subplots(self.predictor_count, 1, figsize=(20, 10))
        
        for i in range(self.predictor_count):
            ax[i].plot(np.float16(self.train_weights[:, i]))
            ax[i].set_title(f"LGBM-Ensemble weight_{i}")
            ax[i].grid(True)
            ax[i].legend()
            
        plt.tight_layout()
        
    def plot_test_weights(self):
        
        fig, ax = plt.subplots(self.predictor_count, 1, figsize=(20, 10))
        
        for i in range(self.predictor_count):
            ax[i].plot(np.float16(self.predict_weights[:, i]))
            ax[i].set_title(f"LGBM-Ensemble weight_{i}")
            ax[i].grid(True)
            ax[i].legend()
            
        plt.tight_layout()
        

