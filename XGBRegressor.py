import numpy as np
import pandas as pd
from XGBTree import *

class XGBRegressor:
    def __init__(self, eta = 0.3, n_estimators = 100, max_depth = 6, gamma = 0, min_child_weight = 1, lambda_ = 1, colsample = 1, subsample = 1):
        self.eta = eta
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.lambda_ = lambda_
        self.colsample = colsample
        self.subsample = subsample
        self.history = {
            "train" : list(),
            "test" : list()
        }
        
        # list of all weak learners
        self.trees = list()
        
        self.base_pred = None
        
        
    
    def fit(self, x, y, eval_set = None):
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
            
            if eval_set:
                X = eval_set[0]
                Y = eval_set[1]
                cost = np.sqrt(np.mean(self.loss(Y, self.predict(X))))
                self.history["test"].append(cost)
                print(f"[{n}] validation_set-rmse : {cost}", end="\t")
            
            cost = np.sqrt(np.mean(self.loss(y, base_pred)))
            self.history["train"].append(cost)
            print(f"[{n}] train_set-rmse : {cost}")
            
    def predict(self, x):
        base_pred = np.full((x.shape[0],), self.base_pred).astype("float64")
        for tree in self.trees:
            base_pred += self.eta * tree.predict(x)
        
        return base_pred
    
    def loss(self, y, a):
        return (y - a)**2
    
    def grad(self, y, a):
        # for 0.5 * (y - a)**2
        return a - y
    
    def hess(self, y, a):
        # for 0.5 * (y - a)**2
        return np.full((y.shape), 1)
    
