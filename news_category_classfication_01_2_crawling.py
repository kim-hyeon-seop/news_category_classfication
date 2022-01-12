from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
import pandas as pd
import re
import time

def crawl_title():
    try:
        title = driver.find_element_by_xpath('//*[@id="section_body"]/ul[{1}]/li[{0}]/dl/dt[2]/a'.format(i, j)).text
        title = re.compile('[^가-힣|a-z|A-Z ]').sub(' ', title)
        print(title)
        titles.append(title)
    except NoSuchElementException:
        title = driver.find_element_by_xpath('//*[@id="section_body"]/ul[{1}]/li[{0}]/dl/dt/a'.format(i, j)).text
        title = re.compile('[^가-힣|a-z|A-Z ]').sub(' ', title)
        print(title)
        titles.append(title)

options = webdriver.ChromeOptions()
options.add_argument('lang=ko_KR')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('disable-gpu')
driver = webdriver.Chrome('./chromedriver',
                          options=options)
df_titles = pd.DataFrame()
pages = [131, 131, 131, 101, 131, 77]
category = ['Politics', 'Economic', 'Social', 'Culture',
            'World', 'IT']
for l in range(6):
    titles = []
    for k in range(1,pages[l]): #406
        url = 'https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=10{}#&date=%2000:00:00&page={}'.format(l, k)
        driver.get(url)
        #time.sleep(0.01)
        for j in range(1,5):
            for i in range(1,6):
                try:
                    crawl_title()
                except StaleElementReferenceException:
                    driver.get(url)
                    print('StaleElementReferenceException')
                    time.sleep(1)
                    crawl_title()
                except:
                    print('error')
        if k % 50 == 0:
            df_section_titles = pd.DataFrame(titles, columns=['title'])
            df_section_titles['category'] = category[l]
            df_section_titles.to_csv('./crawling/news_{}_{}-{}.csv'.format(category[l], k-49, k),
                                     index= False)
            titles = []
    df_section_titles = pd.DataF   rame(titles, columns=['title'])
    df_section_titles['category'] = category[l]
    df_section_titles.to_csv('./crawling/news_{}_remain.csv'.format(category[l]) , index = False)

df_titles.to_csv('./crawling/naver_news.csv')
print(len(titles))
driver.close()

#//*[@id="section_body"]/ul[1]/li[1]/dl/dt[2]/a
#//*[@id="section_body"]/ul[1]/li[2]/dl/dt[2]/a
#//*[@id="section_body"]/ul[1]/li[5]/dl/dt[2]/a
#//*[@id="section_body"]/ul[2]/li[1]/dl/dt[2]/a
#//*[@id="section_body"]/ul[2]/li[1]/dl/dt[1]/a/img
#//*[@id="section_body"]/ul[2]/li[2]/dl/dt/a
#//*[@id="section_body"]/ul[2]/li[3]/dl/dt[2]/a
#//*[@id="section_body"]/ul[3]/li[1]/dl/dt[2]/a
#//*[@id="section_body"]/ul[4]/li[5]/dl/dt[2]/a






