import numpy as np
import pandas as pd
from Node import *
class XGBTree:
    def __init__(self, x, y, grad, hess, depth = 6, gamma = 0, min_child_weight = 1, lambda_ = 1, colsample = 1, subsample = 1):
        indices = np.random.permutation(x.shape[0])[:round(subsample * x.shape[0])]
        
        self.tree = Node(
            x[indices],
            y[indices],
            grad[indices],
            hess[indices],
            depth = depth,
            gamma = gamma,
            min_child_weight = min_child_weight,
            lambda_ =  lambda_,
            colsample = colsample,
        )
    
    def predict(self, x):
        return self.tree.predict(x)
    
