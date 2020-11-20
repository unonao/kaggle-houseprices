import numpy as np
import pandas as pd
import time
import contextlib
from sklearn import metrics


@contextlib.contextmanager
def simple_timer():
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def load_datasets(feats):
    dfs = [pd.read_feather(f'features/{f}_train.feather') for f in feats]
    X_train = pd.concat(dfs, axis=1, sort=False)
    dfs = [pd.read_feather(f'features/{f}_test.feather') for f in feats]
    X_test = pd.concat(dfs, axis=1, sort=False)
    return X_train, X_test


def load_target(target_name):
    train = pd.read_feather(f'data/interim/train.feather')
    y_train = train[target_name]
    return y_train


def evaluate_score(true, predicted, metric_name):
    if metric_name == 'rmse':
        return np.sqrt(metrics.mean_squared_error(true, predicted))


def print_evaluate_regression(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
