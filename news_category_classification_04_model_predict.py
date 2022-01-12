from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
import datetime
import glob
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pickle


pd.set_option('display.unicode.east_asian_width', True)
category = ['Politics', 'Economic', 'Social', 'Culture',
            'World', 'IT']
headers = {"User-Agent":'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36'}
url = 'https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=100'
titles = []

resp = requests.get(url, headers=headers)

soup = BeautifulSoup(resp.text, 'html.parser')
title_tags = soup.select('.cluster_text_headline')



for title_tag in title_tags:
    titles.append(re.compile('[^가-힣|a-z|A-Z ]').sub(' ', title_tag.text))

df_titles = pd.DataFrame()
re_title = re.compile('[^가-힣|a-z|A-Z ]')

for i in range(6):
    resp = requests.get('https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=10{}'.format(i), headers=headers)
    soup = BeautifulSoup(resp.text, 'html.parser')
    title_tags = soup.select('.cluster_text_headline')
    titles = []
    for title_tag in title_tags:
        titles.append(re_title.sub(' ', title_tag.text))
    df_section_titles = pd.DataFrame(titles, columns=['title'])
    df_section_titles['category'] = category[i]
    df_titles = pd.concat([df_titles, df_section_titles],
                axis='rows', ignore_index=True)

df_titles.to_csv('./naver_headline_news{}.csv'.format(
    datetime.datetime.today().strftime('%Y%m%d')))

df = df_titles

X = df['title']
Y = df['category']

encoder = LabelEncoder()
labeled_Y = encoder.transform(Y)    # numpy array로 바뀜
label = encoder.classes_

with open('./models/encoder.pickle', 'rb') as f:
    encoder = pickle.load(f)

onehot_Y = to_categorical(labeled_Y)
