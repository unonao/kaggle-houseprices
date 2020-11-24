import pandas as pd
import numpy as np
import re as re

from base import Feature, get_arguments, generate_features
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
Feature.dir = 'features'


def fixing_skewness(df):
    """
    This function takes in a dataframe and return fixed skewed dataframe
    """

    # Getting all the data that are not of "object" type.
    numeric_feats = df.dtypes[df.dtypes != "object"].index

    # Check the skew of all numerical features
    skewed_feats = df[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)
    high_skew = skewed_feats[abs(skewed_feats) > 0.5]
    skewed_features = high_skew.index

    for feat in skewed_features:
        df[feat] = boxcox1p(df[feat], boxcox_normmax(df[feat] + 1))

    return df


def overfit_reducer(df):
    """
    This function takes in a dataframe and returns a list of features that are overfitted.
    """
    overfit = []
    for i in df.columns:
        counts = df[i].value_counts()
        zeros = counts.iloc[0]
        if zeros / len(df) * 100 > 99:
            overfit.append(i)
    overfit = list(overfit)
    return overfit


class Temporal(Feature):
    '''
    ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MoSold', 'YrSold']
    Nanは GarageYrBlt のみ
    '''

    def create_features(self):
        df = features[temporal_features].copy()

        # feature-eng on temporal-dataset

        for feature in temporal_features:
            if feature == 'YrSold' or feature == 'MoSold':
                pass
            else:
                df[feature + "_diff"] = df['YrSold'] - df[feature]
                df[feature + "_diff"] = df[feature + "_diff"].fillna(df[feature + "_diff"].max())  # 0は良くなってしまうのでmaxで埋める

        df['YrBltAndRemod'] = df['YearBuilt']+df['YearRemodAdd']

        for feature in temporal_features:
            df[feature] = df[feature].astype(str)
            df[feature] = df[feature].fillna("Missing")

        df = pd.get_dummies(df)
        df = df.drop(overfit_reducer(df), axis=1)
        self.train = df[:train.shape[0]]
        self.test = df[train.shape[0]:]


class Objects(Feature):
    def create_features(self):
        df = features[categorical_features].copy()

        # Filling these columns With most suitable value for these columns
        df['Functional'] = df['Functional'].fillna('Typ')

        df['Utilities'] = df['Utilities'].fillna('AllPub')
        df['Electrical'] = df['Electrical'].fillna("SBrkr")
        df['KitchenQual'] = df['KitchenQual'].fillna("TA")

        # Filling these with MODE , i.e. , the most frequent value in these columns .
        df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
        df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
        df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])

        for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
            df[col] = df[col].fillna('None')

        # Same with basement. Missing data in Bsmt most probably means missing basement , so replace NaN with zero .
        for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
            features[col] = features[col].fillna('None')

        #  Idea is that similar MSSubClasses will have similar MSZoning
        features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

        for feature in categorical_features:
            df[feature] = df[feature].fillna("Missing")

        '''
        # label encode
        cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
                'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
                'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
                'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir')
        for c in cols:
            lbl = LabelEncoder()
            lbl.fit(list(df[c].values))
            df[c] = lbl.transform(list(df[c].values))

        # some more-feature engineering:
        df["TotalGarageQual"] = df["GarageQual"] * df["GarageCond"]
        df["TotalExteriorQual"] = df["ExterQual"] * df["ExterCond"]
        '''

        df = pd.get_dummies(df)
        df = df.drop(overfit_reducer(df), axis=1)
        self.train = df[:train.shape[0]]
        self.test = df[train.shape[0]:]


class ObjectsDrop(Feature):
    def create_features(self):
        df = features[categorical_features].copy()

        # Filling these columns With most suitable value for these columns
        df['Functional'] = df['Functional'].fillna('Typ')

        df['Utilities'] = df['Utilities'].fillna('AllPub')
        df['Electrical'] = df['Electrical'].fillna("SBrkr")
        df['KitchenQual'] = df['KitchenQual'].fillna("TA")

        # Filling these with MODE , i.e. , the most frequent value in these columns .
        df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
        df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
        df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])

        for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
            df[col] = df[col].fillna('None')

        # Same with basement. Missing data in Bsmt most probably means missing basement , so replace NaN with zero .
        for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
            features[col] = features[col].fillna('None')

        #  Idea is that similar MSSubClasses will have similar MSZoning
        features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

        for feature in categorical_features:
            df[feature] = df[feature].fillna("Missing")

        # 欠損値が多いもの（40％以上のものを削除）
        drop_col = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"]
        # 欠損値が多いもの（20％以上のものを削除）
        #drop_col = ["PoolQC", "MiscFeature", "Alley", "Fence"]
        df = df.drop(drop_col, axis=1)

        '''
        # label encode
        cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
                'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
                'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
                'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir')
        for c in cols:
            lbl = LabelEncoder()
            lbl.fit(list(df[c].values))
            df[c] = lbl.transform(list(df[c].values))

        # some more-feature engineering:
        df["TotalGarageQual"] = df["GarageQual"] * df["GarageCond"]
        df["TotalExteriorQual"] = df["ExterQual"] * df["ExterCond"]
        '''

        df = pd.get_dummies(df)
        df = df.drop(overfit_reducer(df), axis=1)
        self.train = df[:train.shape[0]]
        self.test = df[train.shape[0]:]


class Numerical(Feature):
    def create_features(self):
        df = features[numeric_features + ["Neighborhood"]].copy()

        # Missing data in GarageYrBit most probably means missing Garage , so replace NaN with zero .
        for col in ('GarageArea', 'GarageCars'):
            df[col] = df[col].fillna(0)

        # fill using neighbor
        for feature in numeric_features:
            df[feature] = df.groupby("Neighborhood")[feature].transform(lambda x: x.fillna(x.median()))
        df = df.drop("Neighborhood", axis=1)

        # feature-generation
        df['TotalHouseSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
        df['TotalLot'] = df['LotFrontage'] + df['LotArea']
        df['TotalBsmtFin'] = df['BsmtFinSF1'] + df['BsmtFinSF2']
        df['TotalBath'] = df['FullBath'] + df['HalfBath']
        df['TotalPorch'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['ScreenPorch']

        df['Total_sqr_footage'] = (df['BsmtFinSF1'] + df['BsmtFinSF2'] +
                                   df['1stFlrSF'] + df['2ndFlrSF'])
        df['Total_Bathrooms'] = (df['FullBath'] + (0.5 * df['HalfBath']) +
                                 df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))
        df['Total_porch_sf'] = (df['OpenPorchSF'] + df['3SsnPorch'] +
                                df['EnclosedPorch'] + df['ScreenPorch'] +
                                df['WoodDeckSF'])
        # For ex, if PoolArea = 0 , Then HasPool = 0 too
        df['haspool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
        df['has2ndfloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
        df['hasgarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
        df['hasbsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
        df['hasfireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

        df = df.drop(overfit_reducer(df), axis=1)
        df = fixing_skewness(df)
        self.train = df[:train.shape[0]]
        self.test = df[train.shape[0]:]


'''
class NewFeatures(Feature):
    def create_features(self):

        c['Year average'] = (c['YearRemodAdd']+c['YearBuilt'])/2
'''


class NumericalEngineering(Feature):
    def create_features(self):
        df = features[numeric_features + ["Neighborhood"]].copy()

        # Missing data in GarageYrBit most probably means missing Garage , so replace NaN with zero .
        for col in ('GarageArea', 'GarageCars'):
            df[col] = df[col].fillna(0)

        # fill using neighbor
        for feature in numeric_features:
            df[feature] = df.groupby("Neighborhood")[feature].transform(lambda x: x.fillna(x.median()))
        df = df.drop("Neighborhood", axis=1)

        # feature-generation
        df['TotalHouseSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

        # TotalArea 重要
        df['TotalArea'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'] + df['GrLivArea'] + df['GarageArea']

        df['Total_Bathrooms'] = (df['FullBath'] + (0.5 * df['HalfBath']) +
                                 df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))
        df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath']

        df['TotalLot'] = df['LotFrontage'] + df['LotArea']
        df['TotalPorch'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['ScreenPorch']

        df['Total_sqr_footage'] = (df['BsmtFinSF1'] + df['BsmtFinSF2'] +
                                   df['1stFlrSF'] + df['2ndFlrSF'])

        df['TotalBsmtFin'] = df['BsmtFinSF1'] + df['BsmtFinSF2']

        df['Total_porch_sf'] = (df['OpenPorchSF'] + df['3SsnPorch'] +
                                df['EnclosedPorch'] + df['ScreenPorch'] +
                                df['WoodDeckSF'])
        # For ex, if PoolArea = 0 , Then HasPool = 0 too
        df['haspool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
        df['has2ndfloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
        df['hasgarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
        df['hasbsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
        df['hasfireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

        df = df.drop(overfit_reducer(df), axis=1)
        df = fixing_skewness(df)
        self.train = df[:train.shape[0]]
        self.test = df[train.shape[0]:]


if __name__ == '__main__':
    args = get_arguments()
    train = pd.read_feather('./data/interim/train.feather')
    test = pd.read_feather('./data/interim/test.feather')
    features = pd.concat([train.drop(['Id', 'SalePrice'], axis=1), test.drop('Id', axis=1)])

    features['MSSubClass'] = features['MSSubClass'].apply(str)

    temporal_features = [feature for feature in features.columns if 'Yr' in feature or 'Year' in feature or 'Mo' in feature]
    numeric_features = [feature for feature in features.columns if features[feature].dtype !=
                        'O' and feature not in temporal_features]
    categorical_features = [feature for feature in features.columns if features[feature].dtype == 'O' and feature not in temporal_features]

    generate_features(globals(), args.force)
