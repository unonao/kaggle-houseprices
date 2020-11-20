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
        df = df[df.GrLivArea < 4500]
        df = df.reset_index(drop=True)
        outliers = [30, 88, 462, 631, 1322]
        df = df.drop(df.index[outliers]).reset_index(drop=True)
    df.to_feather('./data/interim/' + v + '.feather')
