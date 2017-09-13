import os
import sys
import glob
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

def tf(doc):
    vectorizer = CountVectorizer(token_pattern=u'(?u)\\b\\w+\\b')
    features = vectorizer.fit_transform(doc)
    terms = vectorizer.get_feature_names()
    return features, terms

if __name__ == '__main__':
    strpath = '../auto/output_headline.txt'
    articles = []
    with open(strpath,'r') as f:
        for line in f:
            articles.append(line)
    print(articles)
    features, terms = tf(articles)
    print(terms)
    print(features.toarray())
    print('features...')
    print(features.todense().shape)
    F = features.todense()
    print('transform ...')
    F[F>=2] = 1
    print(F);np.save('../auto/output.npy',F)
    
