import logging
#from logs.logger import log_evaluation

import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import ElasticNet, Lasso, Ridge,  BayesianRidge, LassoLarsIC, LinearRegression, RANSACRegressor
from sklearn.pipeline import make_pipeline

from models.base import Model


class LinearRegressionWrapper(Model):
    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, params):
        lr = make_pipeline(RobustScaler(), LinearRegression())
        lr.fit(X_train, y_train)

        # テストデータを予測する
        y_valid_pred = lr.predict(X_valid)
        y_pred = lr.predict(X_test)
        return y_pred, y_valid_pred, lr


class LassoWrapper(Model):
    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, params):
        lr = make_pipeline(RobustScaler(), Lasso(alpha=params['alpha']))
        lr.fit(X_train, y_train)

        # テストデータを予測する
        y_valid_pred = lr.predict(X_valid)
        y_pred = lr.predict(X_test)
        return y_pred, y_valid_pred, lr


class RidgeWrapper(Model):
    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, params):
        lr = make_pipeline(RobustScaler(), Ridge(alpha=params['alpha']))
        lr.fit(X_train, y_train)

        # テストデータを予測する
        y_valid_pred = lr.predict(X_valid)
        y_pred = lr.predict(X_test)
        return y_pred, y_valid_pred, lr


class ElasticNetWrapper(Model):
    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, params):
        lr = make_pipeline(RobustScaler(), ElasticNet(alpha=params['alpha'], l1_ratio=params['l1_ratio']))
        lr.fit(X_train, y_train)

        # テストデータを予測する
        y_valid_pred = lr.predict(X_valid)
        y_pred = lr.predict(X_test)
        return y_pred, y_valid_pred, lr
