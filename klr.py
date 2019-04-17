# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:42:38 2019

@author: vince
"""

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.special import expit
import numpy as np
from sklearn.metrics import accuracy_score

class KernelLogisticRegression(BaseEstimator, ClassifierMixin):

    def __init__(self, C=100, gamma=.01, tol=0.0001, fit_intercept=True, max_iter=20):
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
                
    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False)
        self.classes_ = unique_labels(y)        
        assert (self.classes_[0] == 0), "Label vector must contain at least one 0."
        assert (self.classes_[1] == 1), "Label vector must contain at least one 1."
        
        self.is_fitted_ = True        
        if self.fit_intercept == True:
            X = np.hstack((X, np.ones((X.shape[0],1))))

        self.X_ = X
        self.y_ = y
        
        self.K = rbf_kernel(X, X, gamma=self.gamma)
        self.a = np.zeros(X.shape[0])
        t = 1
        i = 0
        
        while t > self.tol and i < self.max_iter:
            s = expit(self.K @ self.a)
            W = np.diag(s*(1-s))
            
            if np.linalg.cond(W) < 10**15:
                z = self.K @ self.a + np.linalg.solve(W, (y - s))
            else:
                z = self.K @ self.a + np.linalg.lstsq(W, (y - s))[0]
                            
            A = self.K.T @ W @ self.K + (1/self.C)*self.K
            b = self.K.T @ W @ z

            if np.linalg.cond(A) < 10**15:
                a_new = np.linalg.solve(A, b)
            else:
                a_new = np.linalg.lstsq(A, b)[0]

            t = np.max(abs(a_new - self.a))
            self.a = a_new
            i += 1
            
#            if i >= self.max_iter:
#                print('Warning: KLR did not converge')
        
        return self
    
    def add_intercept(self, X):
        return np.hstack((X, np.ones((X.shape[0],1))))

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def predict_proba(self, X):
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        
        if self.fit_intercept == True: 
            X = self.add_intercept(X)

        K = rbf_kernel(X, self.X_, gamma=self.gamma)
        p = expit(K @ self.a)
        return np.column_stack((1-p, p))

    def predict(self, X):
        p = self.predict_proba(X)[:,1]
        return (p > 0.5)*1

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        return {"C": self.C, 
                "gamma": self.gamma,
                "tol": self.tol,
                "fit_intercept": self.fit_intercept}