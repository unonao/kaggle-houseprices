from sklearn.ensemble import RandomForestRegressor
import logging
#from logs.logger import log_evaluation

import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline

from models.base import Model


class RandomForestWrapper(Model):
    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, params):
        reg = make_pipeline(RobustScaler(), RandomForestRegressor(**params))
        reg.fit(X_train, y_train)

        # テストデータを予測する
        y_valid_pred = reg.predict(X_valid)
        y_pred = reg.predict(X_test)
        return y_pred, y_valid_pred, reg
