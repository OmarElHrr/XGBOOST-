import numpy as np
import pandas as pd

class Node:
    def __init__(self, x, y, grad, hess, depth = 6, gamma = 0, min_child_weight = 1, lambda_ = 1, colsample = 1):
        self.x = x
        self.y = y
        self.grad = grad
        self.hess = hess
        self.depth = depth
        self.gamma = gamma
        self.lambda_ = lambda_
        self.min_child_weight = min_child_weight
        self.colsample = colsample
        self.cols = np.random.permutation(x.shape[1])[:round(colsample * x.shape[1])]
        self.sim_score = self.similarity_score([True]*x.shape[0])
        self.gain = float("-inf")
        
        self.split_col = None
        self.split_row = None
        self.lhs_tree = None
        self.rhs_tree = None
        self.pivot = None
        self.val = None
        # making split
        self.split_node()
        
        if self.is_leaf:
            self.val = - np.sum(grad) / (np.sum(hess) + lambda_)
        
    
    def split_node(self):
        
        self.find_split()
        
        # checking whether it's a leaf or not
        if self.is_leaf:
            return
        
        x = self.x[:, self.split_col]
        
        lhs = x <= x[self.split_row]
        rhs = x > x[self.split_row]
        
        # creating further nodes recursivly
        self.lhs_tree = Node(
            self.x[lhs],
            self.y[lhs],
            self.grad[lhs],
            self.hess[lhs],
            depth = self.depth - 1,
            gamma = self.gamma,
            min_child_weight = self.min_child_weight,
            lambda_ = self.lambda_,
            colsample = self.colsample
        )
        
        self.rhs_tree = Node(
            self.x[rhs],
            self.y[rhs],
            self.grad[rhs],
            self.hess[rhs],
            depth = self.depth - 1,
            gamma = self.gamma,
            min_child_weight = self.min_child_weight,
            lambda_ = self.lambda_,
            colsample = self.colsample
        )
        
    def find_split(self):
        # iterate through every feature and row
        for c in self.cols:
            x = self.x[:, c]
            for row in range(self.x.shape[0]):
                pivot= x[row]
                lhs = x <= pivot
                rhs = x > pivot
                sim_lhs = self.similarity_score(lhs)
                sim_rhs = self.similarity_score(rhs)
                gain = sim_lhs + sim_rhs - self.sim_score - self.gamma
                
                if gain < 0 or self.not_valid_split(lhs) or self.not_valid_split(rhs):
                    continue
                
                if gain > self.gain:
                    self.split_col = c
                    self.split_row = row
                    self.pivot = pivot
                    self.gain = gain
                    
    def not_valid_split(self, masks):
        if np.sum(self.hess[masks]) < self.min_child_weight:
            return True
        return False
    
    @property
    def is_leaf(self):
        if self.depth < 0 or self.gain == float("-inf"):
            return True
        return False
                
    def similarity_score(self, masks):
        return  np.sum(self.grad[masks]) ** 2 / ( np.sum(self.hess[masks]) + self.lambda_ )
    
    
    def predict(self, x):
        return np.array([self.predict_single_val(row) for row in x])
    
    def predict_single_val(self, x):
        if self.is_leaf:
            return self.val
        
        return self.lhs_tree.predict_single_val(x) if x[self.split_col] <= self.pivot else self.rhs_tree.predict_single_val(x)
    
    