import pandas as pd
import datetime
import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_log_error
import argparse
import json
import numpy as np

from utils import load_datasets, load_target, evaluate_score
from logs.logger import log_best
from models import LightGBM, LinearRegressionWrapper, LassoWrapper, RidgeWrapper, ElasticNetWrapper, KernelRidgeWrapper, SVRWrapper, XGBoost, RandomForestWrapper, GradientBoostingRegressorWrapper, CatBoost

# 引数で config の設定を行う
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/stacking.json')
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

# FOLDS 数
BASE_FOLDS = 3
META_FOLDS = 3
SK_NUM = 50
seed = 0


def stacking(X_train_all, y_train_all, X_test):
    qcut_target = pd.qcut(y_train_all, SK_NUM, labels=False)

    # 学習前にy_trainに、log(y+1)で変換
    y_train_all = np.log(y_train_all + 1)  # np.log1p() でもOK

    # base model の学習
    base_models = config['base_models']
    # 行数を揃えた空のデータフレームを作成
    oof_df = pd.DataFrame(index=[i for i in range(X_train_all.shape[0])])  # meta model の X_train に
    y_preds_df = pd.DataFrame(index=[i for i in range(X_test.shape[0])])  # meta model の X_test に

    # base model ごとにK-fold して学習
    for name, json_name in base_models.items():
        one_config = json.load(open(f"./configs/{json_name}"))

        oof = np.zeros((X_train_all.shape[0], 1))
        #y_preds = np.zeros((X_test.shape[0], 1))
        y_preds = []
        scores = []
        kf = StratifiedKFold(n_splits=BASE_FOLDS, shuffle=True, random_state=seed)
        for train_index, valid_index in kf.split(X_train_all, qcut_target):
            X_train, X_valid = (X_train_all.iloc[train_index, :], X_train_all.iloc[valid_index, :])
            y_train, y_valid = (y_train_all.iloc[train_index], y_train_all.iloc[valid_index])
            if name == "LightGBM":
                model = LightGBM()
            elif name == "LinearRegression":
                model = LinearRegressionWrapper()
            elif name == "Lasso":
                model = LassoWrapper()
            elif name == "Ridge":
                model = RidgeWrapper()
            elif name == "ElasticNet":
                model = ElasticNetWrapper()
            elif name == "KernelRidge":
                model = KernelRidgeWrapper()
            elif name == "SVR":
                model = SVRWrapper()
            elif name == "XGBoost":
                model = XGBoost()
            elif name == "RandomForest":
                model = RandomForestWrapper()
            elif name == "GradientBoosting":
                model = GradientBoostingRegressorWrapper()
            elif name == "CatBoost":
                model = CatBoost()

            y_pred, y_valid_pred, m = model.train_and_predict(X_train, X_valid, y_train, y_valid, X_test, one_config["params"])

            oof[valid_index, :] = y_valid_pred.reshape(len(y_valid_pred), 1)
            #y_preds += (y_pred / FOLDS)
            y_preds.append(y_pred)
            # スコア
            rmse_valid = evaluate_score(y_valid, y_valid_pred, config['loss'])
            logging.debug(f"\tmodel:{name}, score: {rmse_valid}")
            scores.append(rmse_valid)

        score = sum(scores) / len(scores)
        print('===CV scores===')
        print(f"\tmodel: {name}, scores: {scores}")
        print(f"\tmodel: {name}, score: {score}")
        logging.debug('===CV scores===')
        logging.debug(f"\tmodel: {name}, scores: {scores}")
        logging.debug(f"\tmodel: {name}, score: {score}")

        oof_df[name] = oof
        y_preds_df[name] = sum(y_preds) / len(y_preds)

    # meta model の学習
    y_preds = []
    scores = []
    kf = StratifiedKFold(n_splits=META_FOLDS, shuffle=True, random_state=seed)
    for train_index, valid_index in kf.split(X_train_all, qcut_target):
        X_train, X_valid = (oof_df.iloc[train_index, :], oof_df.iloc[valid_index, :])
        y_train, y_valid = (y_train_all.iloc[train_index], y_train_all.iloc[valid_index])
        name = config['meta_model']
        if name == "LightGBM":
            model = LightGBM()
        elif name == "LinearRegression":
            model = LinearRegressionWrapper()
        elif name == "Lasso":
            model = LassoWrapper()
        elif name == "Ridge":
            model = RidgeWrapper()
        elif name == "ElasticNet":
            model = ElasticNetWrapper()
        elif name == "KernelRidge":
            model = KernelRidgeWrapper()
        elif name == "SVR":
            model = SVRWrapper()
        elif name == "XGBoost":
            model = XGBoost()
        elif name == "RandomForest":
            model = RandomForestWrapper()
        elif name == "GradientBoosting":
            model = GradientBoostingRegressorWrapper()
        elif name == "CatBoost":
            model = CatBoost()

        # 学習と推論。 y_preds_df を X_test に使用する
        y_pred, y_valid_pred, m = model.train_and_predict(X_train, X_valid, y_train, y_valid, y_preds_df, params)

        # 結果の保存
        y_preds.append(y_pred)

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

    stacking(X_train_all, y_train_all, X_test)


if __name__ == '__main__':
    main()
