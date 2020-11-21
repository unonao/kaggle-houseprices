import logging
#from logs.logger import log_evaluation

import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline

from models.base import Model


class KernelRidgeWrapper(Model):
    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, params):

        lr = make_pipeline(RobustScaler(), KernelRidge(alpha=params['alpha'], kernel=params['kernel'], degree=params['degree'], coef0=params['coef0']))
        lr.fit(X_train, y_train)

        # テストデータを予測する
        y_valid_pred = lr.predict(X_valid)
        y_pred = lr.predict(X_test)
        return y_pred, y_valid_pred, lr


class SVRWrapper(Model):
    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, params):

        lr = make_pipeline(RobustScaler(), SVR(kernel=params['kernel'], degree=params['degree'], coef0=params['coef0'], C=params['C'], epsilon=params['epsilon']))
        lr.fit(X_train, y_train)

        # テストデータを予測する
        y_valid_pred = lr.predict(X_valid)
        y_pred = lr.predict(X_test)
        return y_pred, y_valid_pred, lr
