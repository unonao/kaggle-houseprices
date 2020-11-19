import pandas as pd
import numpy as np
import re as re

from base import Feature, get_arguments, generate_features
from sklearn.preprocessing import LabelEncoder
Feature.dir = 'features'


class Numerical(Feature):
    def create_features(self):
        numerical_cols = ['LotFrontage', 'LotArea', 'OverallQual',
                          'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
                          'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
                          'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
                          'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
                          'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
                          'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
                          'MiscVal', 'MoSold', 'YrSold']
        self.train[numerical_cols] = train[numerical_cols]
        self.test[numerical_cols] = test[numerical_cols]


class LabelEncode(Feature):
    def create_features(self):
        cols = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',  'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
                'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
                'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
                ]
        train["PoolQC"] = train["PoolQC"].fillna("None")
        test["PoolQC"] = test["PoolQC"].fillna("None")
        train["Alley"] = train["Alley"].fillna("None")
        test["Alley"] = test["Alley"].fillna("None")

        train["Fence"] = train["Fence"].fillna("None")
        test["Fence"] = test["Fence"].fillna("None")
        train["FireplaceQu"] = train["FireplaceQu"].fillna("None")
        test["FireplaceQu"] = test["FireplaceQu"].fillna("None")

        for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
            train[col] = train[col].fillna('None')
            test[col] = test[col].fillna('None')
        for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
            train[col] = train[col].fillna(0)
            test[col] = test[col].fillna(0)
        for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
            train[col] = train[col].fillna('None')
            test[col] = test[col].fillna('None')
        train["Functional"] = train["Functional"].fillna("Typ")
        test["Functional"] = test["Functional"].fillna("Typ")
        train['KitchenQual'] = train['KitchenQual'].fillna(train['KitchenQual'].mode()[0])
        test['KitchenQual'] = test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])
        train['MSSubClass'] = train['MSSubClass'].fillna("None")
        test['MSSubClass'] = test['MSSubClass'].fillna("None")

        # MSSubClass=The building class
        train['MSSubClass'] = train['MSSubClass'].apply(str)
        test['MSSubClass'] = test['MSSubClass'].apply(str)

        # Changing OverallCond into a categorical variable
        train['OverallCond'] = train['OverallCond'].astype(str)
        test['OverallCond'] = test['OverallCond'].astype(str)

        # process columns, apply LabelEncoder to categorical features
        for c in cols:
            lbl = LabelEncoder()
            lbl.fit(list(train[c].values)+list(test[c].values))
            self.train[c] = lbl.transform(list(train[c].values))
            self.test[c] = lbl.transform(list(test[c].values))


if __name__ == '__main__':
    args = get_arguments()
    train = pd.read_feather('./data/interim/train.feather')
    test = pd.read_feather('./data/interim/test.feather')
    print(train.head())

    generate_features(globals(), args.force)
