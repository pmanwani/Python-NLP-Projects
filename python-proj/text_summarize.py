from urllib.request import urlopen
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.probability import FreqDist
from heapq import nlargest
from collections import defaultdict
articleURL = "https://www.washingtonpost.com/news/the-switch/wp/2016/10/18/the-pentagons-massive-new-telescope-is-designed-to-track-space-junk-and-watch-out-for-killer-asteroids/?noredirect=on&utm_term=.512f0d09f8ba"

page= urlopen(articleURL)
soup = BeautifulSoup(page,'lxml')


text = ''.join(map(lambda r:r.text,soup.findAll('p')))
res1 = soup.find('span',attrs= {'class':'pb-caption'}).text
text+= res1


def summarize(text,n):
    sent = sent_tokenize(text.replace('\xa0', ' '))
    assert n <= len(sent)
    word = word_tokenize(text.lower())
    _stop = set(stopwords.words('english') + list(punctuation))
    word = [w for w in word if w not in _stop]
    freq = FreqDist(word)
    ranking = defaultdict(int)

    for i, s in enumerate(sent):
        for w in word_tokenize(s.lower()):
            if w in freq:
                ranking[i] += freq[w]

    sent_idx = nlargest(n, ranking, key=ranking.get)
    return [sent[j] for j in sorted(sent_idx)]

print(summarize(text,4))