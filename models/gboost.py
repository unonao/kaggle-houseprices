import logging
from logs.logger import log_evaluation, log_evaluation_xgb

import pandas as pd
import lightgbm as lgb
import xgboost as xgb

from models.base import Model


class LightGBM(Model):
    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, params):

        # データセットを生成する
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

        # ロガーの作成
        logger = logging.getLogger('main')
        callbacks = [log_evaluation(logger, period=30)]

        # 上記のパラメータでモデルを学習する
        model = lgb.train(
            params, lgb_train,
            # モデルの評価用データを渡す
            valid_sets=lgb_eval,
            # 最大で 5000 ラウンドまで学習する
            num_boost_round=5000,
            # 10 ラウンド経過しても性能が向上しないときは学習を打ち切る
            early_stopping_rounds=10,
            # ログ
            callbacks=callbacks
        )

        # valid を予測する
        y_valid_pred = model.predict(X_valid, num_iteration=model.best_iteration)
        # テストデータを予測する
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)

        return y_pred, y_valid_pred, model


class XGBoost(Model):
    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, params):

        dtrain = xgb.DMatrix(X_train, y_train)
        deval = xgb.DMatrix(X_valid, y_valid)

        # specify validations set to watch performance
        watchlist = [(deval, 'eval'), (dtrain, 'train')]

        # ロガーの作成
        logger = logging.getLogger('main')
        callbacks = [log_evaluation_xgb(logger, period=30, show_stdv=True)]

        # 上記のパラメータでモデルを学習する
        model = xgb.train(
            params, dtrain,
            # 最大で 5000 ラウンドまで学習する
            num_boost_round=5000,
            evals=watchlist,
            # 10 ラウンド経過しても性能が向上しないときは学習を打ち切る
            early_stopping_rounds=10,
            # ログ
            callbacks=callbacks
        )

        # valid を予測する
        y_valid_pred = model.predict(deval)
        # テストデータを予測する
        dtest = xgb.DMatrix(X_test)
        y_pred = model.predict(dtest)

        return y_pred, y_valid_pred, model
