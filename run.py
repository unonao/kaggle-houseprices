import pandas as pd
import datetime
import logging
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
import argparse
import json
import numpy as np

from utils import load_datasets, load_target, evaluate_score
from logs.logger import log_best
from models import LightGBM, LinearRegressionWrapper, LassoWrapper, RidgeWrapper

# 引数で config の設定を行う
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.json')
options = parser.parse_args()
config = json.load(open(options.config))

feats = config['features']
target_name = config['target_name']
params = config['params']

# log の設定
now = datetime.datetime.now()
logging.basicConfig(
    filename='./logs/log_{0}_{1:%Y%m%d%H%M%S}.log'.format(config['model'], now), level=logging.DEBUG
)


def train_and_predict_lightgbm(X_train_all, y_train_all, X_test):

    # 学習前にy_trainに、log(y+1)で変換
    y_train_all = np.log(y_train_all + 1)  # np.log1p() でもOK

    # グリッドサーチし、学習する。

    y_preds = []
    models = []
    kf = KFold(n_splits=5)
    for train_index, valid_index in kf.split(X_train_all):
        X_train, X_valid = (X_train_all.iloc[train_index, :], X_train_all.iloc[valid_index, :])
        y_train, y_valid = (y_train_all.iloc[train_index], y_train_all.iloc[valid_index])

        # lgbmの実行
        lgbm = LightGBM()
        y_pred, y_valid_pred, model = lgbm.train_and_predict(X_train, X_valid, y_train, y_valid, X_test, params)

        # 結果の保存
        y_preds.append(y_pred)
        models.append(model)

        # スコア
        log_best(model, config['loss'])

    # CVスコア
    scores = [
        m.best_score['valid_0'][config['loss']] for m in models
    ]
    score = sum(scores) / len(scores)
    print('===CV scores===')
    print(scores)
    print(score)
    logging.debug('===CV scores===')
    logging.debug(scores)
    logging.debug(score)

    # submitファイルの作成
    ID_name = config['ID_name']
    sub = pd.DataFrame(pd.read_feather(f'data/interim/test.feather')[ID_name])

    y_sub = sum(y_preds) / len(y_preds)

    # 最後に、予測結果に対しexp(y)-1で逆変換
    y_sub = np.exp(y_sub) - 1  # np.expm1() でもOK

    sub[target_name] = y_sub

    sub.to_csv(
        './data/output/sub_{0:%Y%m%d%H%M%S}_{1}.csv'.format(now, score),
        index=False
    )


def train_and_predict_linear(X_train_all, y_train_all, X_test):

    # 学習前にy_trainに、log(y+1)で変換
    y_train_all = np.log(y_train_all + 1)  # np.log1p() でもOK

    # グリッドサーチし、学習する。

    y_preds = []
    models = []
    # CVスコア
    scores = []
    kf = KFold(n_splits=5)
    for train_index, valid_index in kf.split(X_train_all):
        X_train, X_valid = (X_train_all.iloc[train_index, :], X_train_all.iloc[valid_index, :])
        y_train, y_valid = (y_train_all.iloc[train_index], y_train_all.iloc[valid_index])

        if config['model'] == "LinearRegression":
            lr = LinearRegressionWrapper()
        elif config['model'] == "Lasso":
            lr = LassoWrapper()
        elif config['model'] == "Ridge":
            lr = RidgeWrapper()
        y_pred, y_valid_pred, model = lr.train_and_predict(X_train, X_valid, y_train, y_valid, X_test, params)

        # 結果の保存
        y_preds.append(y_pred)
        models.append(model)

        # スコア
        rmse_valid = evaluate_score(y_valid, y_valid_pred, config['loss'])
        logging.debug(f"\tscore: {rmse_valid}")
        scores.append(rmse_valid)

    score = sum(scores) / len(scores)
    print('===CV scores===')
    print(scores)
    print(score)
    logging.debug('===CV scores===')
    logging.debug(scores)
    logging.debug(score)

    # submitファイルの作成
    ID_name = config['ID_name']
    sub = pd.DataFrame(pd.read_feather(f'data/interim/test.feather')[ID_name])

    y_sub = sum(y_preds) / len(y_preds)

    # 最後に、予測結果に対しexp(y)-1で逆変換
    y_sub = np.exp(y_sub) - 1  # np.expm1() でもOK

    sub[target_name] = y_sub

    sub.to_csv(
        './data/output/sub_{0}_{1:%Y%m%d%H%M%S}_{2}.csv'.format(config['model'], now, score),
        index=False
    )


def main():
    logging.debug('./logs/log_{0:%Y%m%d%H%M%S}.log'.format(now))

    logging.debug('config: {}'.format(options.config))
    logging.debug(feats)
    logging.debug(params)

    # 指定した特徴量からデータをロード
    X_train_all, X_test = load_datasets(feats)
    y_train_all = load_target(target_name)
    logging.debug("X_train_all shape: {}".format(X_train_all.shape))
    if config['model'] == 'LightGBM':
        train_and_predict_lightgbm(X_train_all, y_train_all, X_test)
    elif config['model'] in ['LinearRegression', 'Lasso', 'Ridge']:
        train_and_predict_linear(X_train_all, y_train_all, X_test)


if __name__ == '__main__':
    main()
