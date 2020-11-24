# kaggle-houseprices

# やること

# 効いたこと

- 外れ値の扱い
  - 意外とテストデータにも外れ値に近い要素がいそう
  - 下手な除外をすると、CV は良くなる（それはそう）が、LB のスコアは悪化
- CV の切り方
  - 外れ値も上手く分割したい
  - stratified k-fold で、予測値の分布に偏りがでないように切る
  - ただし、データセットが少ないので、あまり信用できないのが難しい（オーバーフッティングかもしれない）
- 特徴量について
  - 他のノートブックも参考にしつつ、なるべく良い特徴量を加えるべき
  - 単純なモデル（線形モデルなど）を使用しているので、モデルの気持ちになって特徴量を作る
- パラメータチューニングについて
  - オーバーフッティングしない程度にチューニング
  - 勾配 Boosting は学習率を下げて十分にイテレーションさせる

# Structures

```
.
├── configs
├── data
│   ├── input
│   ├── interim
│   └── output
├── features
├── logs
├── models
├── notebooks
├── scripts
├── utils
├── .gitignore
├── README.md
└── run.py

```

## configs

実験ごとに利用している特徴量とパラメータの管理をする。
json ファイルで記載。

## data

コンペのデータ置き場。
input は操作しない。
output は出力結果を保存するだけ。
形式は `sub_(year-month-day-hour-min)_(score).csv`

## features

自分で生成した特徴量諸々
create.py に作成したクラスを元に生成される

## logs

ログの結果。
形式は `log_(year-month-day-hour-min)_(score).log`
提出用の csv ファイルと照合できるようにする。

- 利用した特徴量
- train の shape
- 学習機のパラメータ
- cv のスコア

## models

学習機のフォルダ。別のコンペでも使い回せることを意識して入出力を構築

- 入力 dataframe, prameter
- 出力 予測結果

## notebook

試行錯誤するための notebook
ここで試行錯誤した結果を適切なフォルダの python ファイルに取り込む

## scripts

汎用的な python ファイルを配置

## utils

汎用的な python 関数を配置し、呼び出せるように

# フォルダ構成の参考

- flowlight さん [優勝したリポジトリ](https://github.com/flowlight0/talkingdata-adtracking-fraud-detection)
- u++さん [【Kaggle のフォルダ構成や管理方法】タイタニック用の GitHub リポジトリを公開しました](https://upura.hatenablog.com/entry/2018/12/28/225234)
- amaotone さん [Kaggle で使える Feather 形式を利用した特徴量管理法](https://amalog.hateblo.jp/entry/kaggle-feature-management)
- amaotone さん[LightGBM の callback を利用して学習履歴をロガー経由で出力する](https://amalog.hateblo.jp/entry/lightgbm-logging-callback)
