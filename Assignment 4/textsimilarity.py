'''
Try alternative to mean-word2vec approaches for word disambiguation using e.g. measures
of text similarity given last time by Tommi in his tutorial with gensim.
'''
import nltk
import string
import re
import nltk
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.util import ngrams
from collections import Counter
from collections import OrderedDict
from nltk import FreqDist
from nltk import CFG
import gzip
import gensim 
import logging

def processdata(path):
    with open (path, 'rb') as f:
        for i, line in enumerate (f): 
            yield gensim.utils.simple_preprocess(line)

def main():
    path = "/Users/Karina/Desktop/NLPCourse/NLP/feature-specific/Dataset2.txt"
    documents = list(processdata(path))
    model = gensim.models.Word2Vec (documents, size=150, window=10, min_count=2, workers=10)
    model.train(documents,total_examples=len(documents),epochs=10)
    word1 = "customer"
    # top 10 similar words
    print("top 10 similar words ")
    print(model.wv.most_similar(positive=word1))
    print("top 6 similar words ")
    word1 = ["contact"]
    print(model.wv.most_similar (positive=word1,topn=6))
    # relation to installation
    word1 = ["installation",'components','product']
    word2 = ['software']
    # similarity between related words
    print(model.wv.most_similar (positive=word1,negative=word2,topn=10))
    # similarity between identical words
    print(model.wv.similarity(w1="customers",w2="customers"))
    # similarity between unrelated words
    print(model.wv.similarity(w1="high",w2="low"))
    # determining odd relation 
    print(model.wv.doesnt_match(["customers","clients","fourth"]))
    
if __name__ == "__main__":
    main()