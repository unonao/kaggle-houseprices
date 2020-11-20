import pandas as pd
import numpy as np
import re as re

from base import Feature, get_arguments, generate_features
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
Feature.dir = 'features'


def fillna_numerical(features):
    # (numerical) Fill the remaining columns as None
    numerics = features.dtypes[features.dtypes != "object"].index
    features[numerics] = features[numerics].fillna(0)
    return features


def fillna_categorical(features):
    features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))

    # Filling these columns With most suitable value for these columns
    features['Functional'] = features['Functional'].fillna('Typ')

    features['Utilities'] = features['Utilities'].fillna('AllPub')
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
    features['MSSubClass'] = features['MSSubClass'].astype(str)
    features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

    features['YrSold'] = features['YrSold'].astype(str)
    features['MoSold'] = features['MoSold'].astype(str)

    # (object) Fill the remaining columns as None

    objects = features.dtypes[features.dtypes == "object"].index
    features[objects] = features[objects].fillna('None')

    return features


def create_features(features):
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
    return features


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


def overfit_reducer(df):
    """
    This function takes in a dataframe and returns a list of features that are overfitted.
    """
    overfit = []
    for i in df.columns:
        counts = df[i].value_counts()
        zeros = counts.iloc[0]
        if zeros / len(df) * 100 > 94.2:
            overfit.append(i)
    overfit = list(overfit)
    return overfit


class ObjectFeatures(Feature):
    def create_features(self):
        objects = features.dtypes[features.dtypes == "object"].index
        final_features = pd.get_dummies(features[objects])
        self.train = final_features[:train.shape[0]]
        self.test = final_features[train.shape[0]:]


class NumericalFeatures(Feature):
    def create_features(self):
        numerics = features.dtypes[features.dtypes != "object"].index
        self.train[numerics] = features[:train.shape[0]][numerics]
        self.test[numerics] = features[train.shape[0]:][numerics]


if __name__ == '__main__':
    args = get_arguments()
    train = pd.read_feather('./data/interim/train.feather')
    test = pd.read_feather('./data/interim/test.feather')
    features = pd.concat([train.drop(['Id', 'SalePrice'], axis=1), test.drop('Id', axis=1)])

    # Removing features that are not very useful . This can be understood only by doing proper EDA on data
    features = fillna_numerical(features)
    features = fillna_categorical(features)
    features = create_features(features)
    features = features.drop(['Utilities', 'Street', 'PoolQC', ], axis=1)
    fixing_skewness(features)

    overfitted_features = overfit_reducer(features[:train.shape[0]])
    print(overfitted_features)
    features = features.drop(overfitted_features, axis=1)
    generate_features(globals(), args.force)
