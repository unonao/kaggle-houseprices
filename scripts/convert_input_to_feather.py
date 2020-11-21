"""
    Firstly, we should convert train&test data.
"""
import pandas as pd

target = {
    'train': 'train',
    'test': 'test',
}

extension = 'csv'
# extension = 'tsv'
# extension = 'zip'

for k, v in target.items():
    df = pd.read_csv('./data/input/' + k + '.' + extension, encoding="utf-8")
    # deleting outliers
    if k == 'train':
        ''' https://www.kaggle.com/amiiiney/price-prediction-regularization-stacking#5--Outliers-detection:
            # GrLivArea 523, 1298
            # TotalBsmtSF  1298
            # MasVnrArea 297
            # 1stFlrSF 1298
            # GarageArea 1298, 581 1190 1061
            # TotRmsAbvGrd 635
        '''
        '''
        df = df[df.GrLivArea < 4500]
        df = df.reset_index(drop=True)
        outliers = [1298]
        df = df.drop(df.index[outliers]).reset_index(drop=True)
        '''
    df.to_feather('./data/interim/' + v + '.feather')
