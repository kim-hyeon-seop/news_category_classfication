import pandas as pd
import numpy as np
import glob


data_paths = glob.glob('./datasets/*.csv')
df = pd.DataFrame()

for data_path in data_paths:
    df_temp = pd.read_csv(data_path, index_col = 0)
    df = pd.concat([df, df_temp])
df.dropna(inplace = True)
df.reset_index(drop = True, inplace = True)

print(df.head())
print(df.tail())
print(df['category'].value_counts())
print(df.info())
df.to_csv('./datasets/naver_news.csv', index = False)


# df = pd.read_csv('./news_Culture_1-50.csv' , index_col = 0) ## 인덱스 컬럼
# df_temp = pd.read_csv('./news_Culture_51-100.csv', index_col = 0)
#
#

