import numpy as np
import pandas as pd
from XGBTree import *


class XGBClassifierBase:
    def __init__(self, eta = 0.3, n_estimators = 100, max_depth = 6, gamma = 0, min_child_weight = 1, lambda_ = 1, colsample = 1, subsample = 1):
        self.eta = eta
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.lambda_ = lambda_
        self.colsample = colsample
        self.subsample = subsample
        # list of all weak learners
        self.trees = list()
        
        self.base_pred = None
    
    def fit(self, x, y):
        # checking Datatypes
        if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            x = x.values
        if not isinstance(x, np.ndarray):
            raise TypeError("Input should be pandas Dataframe/Series or numpy array.")
            
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
        if not isinstance(y, np.ndarray):
            raise TypeError("Input should be pandas Dataframe/Series or numpy array.")
        
        
        
        base_pred = np.full(y.shape, np.mean(y)).astype("float64")
        self.base_pred = np.mean(y)
        for n in range(self.n_estimators):
            grad = self.grad(y, base_pred)
            hess = self.hess(y, base_pred)
            estimator = XGBTree(
                x,
                y,
                grad,
                hess,
                depth = self.max_depth,
                gamma = self.gamma,
                min_child_weight = self.min_child_weight,
                lambda_ = self.lambda_,
                colsample = self.colsample,
                subsample = self.subsample
            )
            base_pred = base_pred + self.eta * estimator.predict(x)
            self.trees.append(estimator)
            
            
    def predict(self, x, prob=True):
        base_pred = np.full((x.shape[0],), self.base_pred).astype("float64")
        for tree in self.trees:
            base_pred += self.eta * tree.predict(x)
        pred_prob = self.sigmoid(base_pred)
        if prob: return pred_prob
        return np.where(pred_prob > 0.5, 1, 0)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def loss(self, y, a):
        return - (y * np.log(a) + (1 - y) * np.log(1 - a))
    
    def grad(self, y, a):
        a_prob = self.sigmoid(a)
        return a_prob - y
    
    def hess(self, y, a):
        a_prob = self.sigmoid(a)
        return a_prob * (1 - a_prob)
    
