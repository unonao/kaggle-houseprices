import pandas as pd
import numpy as np
import re as re

from base import Feature, get_arguments, generate_features
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
Feature.dir = 'features'


def fillna_categorical(features):
    # Filling these columns With most suitable value for these columns
    features['Functional'] = features['Functional'].fillna('Typ')
    features['Electrical'] = features['Electrical'].fillna("SBrkr")
    features['KitchenQual'] = features['KitchenQual'].fillna("TA")

    # Filling these with MODE , i.e. , the most frequent value in these columns .
    features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])
    features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
    features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])

    # Missing data in GarageYrBit most probably means missing Garage , so replace NaN with zero .
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        features[col] = features[col].fillna(0)

    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        features[col] = features[col].fillna('None')

    # Same with basement. Missing data in Bsmt most probably means missing basement , so replace NaN with zero .

    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        features[col] = features[col].fillna('None')

    #  Idea is that similar MSSubClasses will have similar MSZoning
    features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

    # (object) Fill the remaining columns as None
    objects = []
    for i in features.columns:
        if features[i].dtype == object:
            objects.append(i)
    features.update(features[objects].fillna('None'))

    return features


def fillna_numerical(features):
    # (numerical) Fill the remaining columns as None
    features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerics = []
    for i in features.columns:
        if features[i].dtype in numeric_dtypes:
            numerics.append(i)
    features.update(features[numerics].fillna(0))

    # change distribution using box cox
    skew_features = features[numerics].apply(lambda x: skew(x)).sort_values(ascending=False)

    high_skew = skew_features[skew_features > 0.5]
    skew_index = high_skew.index

    for i in skew_index:
        features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))

    return features


class ObjectFeatures(Feature):
    def create_features(self):
        objects = []
        for i in features.columns:
            if features[i].dtype == object:
                objects.append(i)
        final_features = pd.get_dummies(features[objects])

        self.train = final_features[:train.shape[0]]
        self.test = final_features[train.shape[0]:]


class NumericalFeatures(Feature):
    def create_features(self):
        numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        numerics = []
        for i in features.columns:
            if features[i].dtype in numeric_dtypes:
                numerics.append(i)
        self.train[numerics] = features[:train.shape[0]][numerics]
        self.test[numerics] = features[train.shape[0]:][numerics]


class NewNumerical(Feature):
    def create_features(self):
        cols = ['YrBltAndRemod', 'TotalSF', 'Total_sqr_footage', 'Total_Bathrooms', 'Total_porch_sf', 'haspool', 'has2ndfloor', 'hasgarage', 'hasbsmt', 'hasfireplace']
        # Adding new features . Make sure that you understand this.
        features['YrBltAndRemod'] = features['YearBuilt']+features['YearRemodAdd']
        features['TotalSF'] = features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']
        features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +
                                         features['1stFlrSF'] + features['2ndFlrSF'])
        features['Total_Bathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) +
                                       features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))
        features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +
                                      features['EnclosedPorch'] + features['ScreenPorch'] +
                                      features['WoodDeckSF'])
        # For ex, if PoolArea = 0 , Then HasPool = 0 too
        features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
        features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
        features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
        features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
        features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

        self.train[cols] = features[:train.shape[0]][cols]
        self.test[cols] = features[train.shape[0]:][cols]


if __name__ == '__main__':
    args = get_arguments()
    train = pd.read_feather('./data/interim/train.feather')
    test = pd.read_feather('./data/interim/test.feather')
    features = pd.concat([train.drop(['Id', 'SalePrice'], axis=1), test.drop('Id', axis=1)])

    # Removing features that are not very useful . This can be understood only by doing proper EDA on data
    features = features.drop(['Utilities', 'Street', 'PoolQC', ], axis=1)
    features = fillna_numerical(features)
    features = fillna_categorical(features)
    print(features.shape)

    generate_features(globals(), args.force)
