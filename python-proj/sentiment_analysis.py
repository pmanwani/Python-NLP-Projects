# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 21:18:39 2018

@author: Prerna
"""
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from urllib.request import urlopen
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
import time
from pandas import DataFrame
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem.wordnet import WordNetLemmatizer



browser = webdriver.Chrome()
url = 'https://www.capitalone.com/bank/reviews/?prodid=checking'
browser.get(url)
reviews= []
i = 0
while i<10 :
    i+=1    
    try:
        
        html = browser.execute_script("return document.body.innerHTML")
        soup = BeautifulSoup(html,'html.parser')
        for review in soup.find_all('div',attrs = {"class":"review-contents"}):
            reviews += map(lambda r: r.text,review.find_all('p'))
        next_link = browser.find_element_by_xpath('//*[@id="page-body-wrapper"]/div/div[3]/div[1]/div/div[2]/div/div[8]/div/button[2]')
        next_link.click()
        time.sleep(30)
    except NoSuchElementException:
        break
browser.close()

_stopwords = set(stopwords.words('english')+ list(punctuation)+[ "capital"+ "one"])
lemma = WordNetLemmatizer()
def clean(doc):
    stopfree = " ".join([i for i in doc.lower().split() if i not in _stopwords])
    normalized = " ".join(lemma.lemmatize(word) for word in stopfree.split())
    return normalized

review_clean = []
for doc in reviews:
    review_clean.append(clean(doc))


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
cvector = CountVectorizer(stop_words = 'english')
cvectorout = cvector.fit_transform(review_clean)
cvectorout.shape
cvector.vocabulary_

tfidf = TfidfVectorizer(analyzer = 'word', min_df = 2, max_df = 0.5, stop_words = 'english')
tfidfmatrix = tfidf.fit_transform(review_clean)
tfidf.get_feature_names()


from sklearn.cluster import KMeans
km= KMeans(n_clusters=3, init='k-means++',max_iter= 100, n_init=1,verbose= True)
km.fit(tfidfmatrix)

text = {} 

for i, cluster in enumerate(km.labels_):
    onedocument = review_clean[i]
    if cluster not in text.keys():
        text[cluster] = onedocument
    else:
        text[cluster]+= onedocument
        
        
        
from heapq import nlargest
from nltk.probability import FreqDist
counts = {}
keywords = {}
for cluster in range(3):
    word_token = word_tokenize(text[cluster].lower())
    word_token = [word for word in word_token if word not in _stopwords]
    freq = FreqDist(word_token)
    keywords[cluster] = nlargest(10,freq,key= freq.get)
    counts[cluster] = freq
    
unique_keys = {}
for cluster in range(3):
    other_clusters = list(set(range(3)) - set([cluster]))
    keys_other = set(keywords[other_clusters[0]]).union(set(keywords[other_clusters[1]]))
    unique = set(keywords[cluster]) - keys_other
    unique_keys[cluster] = nlargest(30,unique,key= counts[cluster].get)
    
#%%

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

analyzer =  SentimentIntensityAnalyzer()
vs_compound = []
vs_pos = []
vs_neg = []
vs_neu = []


for i in range(0,len(reviews)):
    vs_compound.append(analyzer.polarity_scores(reviews[i])['compound']) 
    vs_pos.append(analyzer.polarity_scores(reviews[i])['pos'])
    vs_neg.append(analyzer.polarity_scores(reviews[i])['neg'])
    vs_neu.append(analyzer.polarity_scores(reviews[i])['neu'])

   
reviews_df = DataFrame({'Review':reviews,'compound':vs_compound,'pos':vs_pos,'neg':vs_neg,'neu':vs_neu})

comp_review = sum(reviews_df['compound']>=0.8)
pos_review = sum(reviews_df['pos']>=0.1)
neg_review = sum(reviews_df['compound']<=-0.1)

sentiment = {'Very Positive': comp_review,'Positive':pos_review,'Negative':neg_review}
sort_senti = sorted(sentiment.items())
plt.bar(range(len(sentiment)),sentiment.values(), align = 'center')
plt.xticks(range(len(sentiment)),sentiment.keys())

#Density Plots

f,axr = plt.subplots(2,2)
f.tight_layout()
axr[0,0].hist(vs_pos)
axr[0,0].set_title('Positive')
axr[0,1].hist(vs_neg)
axr[0,1].set_title('Negative')
axr[1,0].hist(vs_neu)
axr[1,0].set_title('Neutral')
axr[1,1].hist(vs_compound)
axr[1,1].set_title('Compound')

#bag of words

from wordcloud import WordCloud
positive = reviews_df[reviews_df['compound']>=0.8]['Review']
negative = reviews_df[reviews_df['compound']<=-0.1]['Review']

df = pd.DataFrame(positive).reset_index()
df.columns = ['number','review']
wordcloud2 = WordCloud().generate(' '.join(df['review']))
plt.imshow(wordcloud2)
plt.axis("off")
plt.show()



df1 = pd.DataFrame(negative).reset_index()
df1.columns = ['number','review']
wordcloud3 = WordCloud().generate(' '.join(df1['review']))
plt.imshow(wordcloud3)
plt.axis("off")
plt.show()
