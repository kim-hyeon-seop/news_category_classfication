import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pickle

# from keras.utils.np_utils import to_categorical
#판다스의 겟더미스랑 똑같이 희소행렬 만들어줌
# ko는 한국어 nlp 는 자연어 처리(이건 콘다 명령어, 인터프리터에서도
# 안먹힘 그래서 pip install 을 이용해서 깔아야함
# 자바는 무조건 패키지 8버전으로(11버전은 호환성이 꾸짐)
# konlp 는 자바언어로 되어있어서 자바를 깔아야함




pd.set_option('display.unicode.east_asian_width', True)
df = pd.read_csv('./datasets/naver_news.csv')
print(df.head())
print(df.info())

X = df['title']
Y = df['category']

encoder = LabelEncoder()
labeled_Y = encoder.fit_transform(Y)    # numpy array로 바뀜
label = encoder.classes_

with open('./models/encoder.pickle', 'wb') as f:
    pickle.dump(encoder, f)

onehot_Y = to_categorical(labeled_Y)
print(onehot_Y)

# 자연어 전처리가 제일 중요한 부분
# 이제 형태소 분리를 시작해야함

okt = Okt()
okt_morph_x = okt.morphs(X[2], stem = True)  # 어절로 쪼개버림 , stem = true 면 어간만 뽑아낸다.

okt_nouns_x = okt.nouns(X[0])
print(X[0])
print(okt_nouns_x)  ## okt 가 완벽하지가 않음

for i in range(len(X)):
    X[i] = okt.morphs(X[i])

stopwords = pd.read_csv('./stopwords.csv', index_col = 0)  # 0번 컬럼을 인덱스로 쓰겠단 이야기
# 폴더 안에 stopword는 학습에 도움안되는 불용어들(감탄사, 대명사 등등)의 csv 파일



for j in range(len(X)):
    words = []
    for i in range(len(X[j])):
        if len(X[j][i]) > 1:
            if X[j][i] not in list(stopwords['stopword']):
                words.append(X[j][i])
    X[j] = ' '.join(words)

print(X)

# 이제 모든 필터링된 어절들을 숫자로 치환하는 작업을 할거임 ( 이 작업이 토크나이저라는 작업 )
# 프리프로세싱.text 안에 토크나이저가 들어있음

token = Tokenizer()
token.fit_on_texts(X)
tokened_X = token.texts_to_sequences(X)
print(tokened_X[:5])

with open('./models/news_token.pickle', 'wb') as f:
    pickle.dump(token, f)

wordsize = len(token.word_index) + 1  # 우리는 0도 쓸거임
print(wordsize)
print(token.index_word)

max = 0

for i in range(len(tokened_X)):
    if max < len(tokened_X[i]):
        max = len(tokened_X[i])
print(max)

X_pad = pad_sequences(tokened_X, max)
print(X_pad[:10])

# max 사이즈에 맞춰서 토큰사이즈를 맞춰줘라
# input_dim 을 맞춰줘야함 이제부터

X_train, X_test, Y_train, Y_test = train_test_split(
    X_pad, onehot_Y, test_size = 0.1
)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

xy = X_train, X_test, Y_train, Y_test
np.save('./models/news_data_max_{}_size_{}'.format(max, wordsize), xy)

# 데이터 전처리는 이게 끝
