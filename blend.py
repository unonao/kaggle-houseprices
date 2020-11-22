import numpy as np
import pandas as pd
ID_name = "Id"
target_name = "SalePrice"

sub = pd.DataFrame(pd.read_feather(f'data/interim/test.feather')[ID_name])
sub[target_name] = 0


base_subs = {
    "cat": "data/output/sub_CatBoost_20201122181954_0.12333458407126845.csv",
    "elastic": "data/output/sub_ELasticNet_20201122182144_0.12740052041172534.csv",
    "gb": "data/output/sub_GradientBoosting_20201122182203_0.12691619533179993.csv",
    "lasso": "data/output/sub_Lasso_20201122182412_0.12735324161273576.csv",
    "lgbm": "data/output/sub_LightGBM_20201122182425_0.1265033894901618.csv",
    "xgb": "data/output/sub_XGBoost_20201122182607_0.12749830372143028.csv",
    "stack": "data/output/sub_stacking_20201122183325_0.11987117086227543.csv",
    #    "rf": "data/output/sub_RandomForest_20201122182446_0.14382690434655043.csv",
    #    "kr": "data/output/sub_KernelRidge_20201122182356_0.1297173732291673.csv",
    #   "ridge": "data/output/sub_Ridge_20201122182514_0.129362827976269.csv",
}

for base, path in base_subs.items():
    tmp_sub = pd.read_csv(path)
    if base == "stack":
        sub[target_name] += tmp_sub[target_name] * 10.
    else:
        sub[target_name] += tmp_sub[target_name]
sub[target_name] /= 16.

sub.to_csv(
    './data/output/sub_blend.csv',
    index=False
)
