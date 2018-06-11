'''
Speech and Language Processing. Daniel Jurafsky & James H. Martin. 
Copyright 2016. All rights reserved. Draft of August 7, 2017.

Language Modeling with N- grams

tasks 4.8 - 4.11
'''
import nltk
import string
import re
import nltk
import random
# split into words
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.util import ngrams
from collections import Counter
from collections import OrderedDict
from nltk import FreqDist
from nltk import CFG

# reading the data from the dataset
def load_data(path):
    print("\n")
    print("Working with a dataset ", path)
    file = open(path, 'rt')
    data = file.read()
    file.close()
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(data)
    print("Length of tokens ", len(tokens))
    return tokens

# pre-process data regarding punctuation, stop-words, etc.
def process_data(data):
    tokens = tokenizer.tokenize(data)    
    # Consider getting rid of punctuation using NLTK tokenizer
    punctuations = string.punctuation
    #print(punctuations)
    stripped = [word for word in tokens if word.isalpha()]
    
    # Normalizing cases - converting all words to one case
    lower_case = [word.lower() for word in stripped]
    
    # Remove remaining tokens that are not alphabetic
    words = [word for word in lower_case if word.isalpha()]

    # Filtering the stop-words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]

    stemmer = SnowballStemmer('english')
    stemming = list(map(stemmer.stem, tokens))
    #print("Stemming results:", stemming)
    
    lemmatiser = WordNetLemmatizer()
    lemmatizing = [lemmatiser.lemmatize(word) for word in words]
    #print("Lemmatization results:", lemmatizing)
    
    unigram = ngrams(lemmatizing,1)
    #print("Unigrams ", Counter(unigram))
    bigram = ngrams(lemmatizing,2)
    #print("Bigrams ", Counter(bigram))

    return stemming

# process labels/features
def process(path):
    label_feats = []  
    with open(path, mode='r') as f:
        for line in f.readlines():
            if "$ neg $" in line:
                labels = "neg"
                tmp = [x.strip() for x in line.split("$ neg $")]
                temp = process_data(tmp[1])
            if "$ pos $" in line:
                labels = "pos"
                tmp = [x.strip() for x in line.split("$ pos $")]
                temp = process_data(tmp[1])
            label_feats.append((labels, temp))
    return label_feats


def process_data(tokens):
    # Consider getting rid of punctuation using NLTK tokenizer
    punctuations = string.punctuation
    #print(punctuations)
    stripped = [word for word in tokens if word.isalpha()]
    
    # Normalizing cases - converting all words to one case
    lower_case = [word.lower() for word in stripped]
    
    # Remove remaining tokens that are not alphabetic
    words = [word for word in lower_case if word.isalpha()]

    # Filtering the stop-words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    
    stemmer = SnowballStemmer('english')
    stemming = set([stemmer.stem(word) for word in words])
    #print("Stemming results:", stemming)
    
    lemmatiser = WordNetLemmatizer()
    lemmatizing = [lemmatiser.lemmatize(word) for word in words]
    #print("Lemmatization results:", lemmatizing)
    
    unigram = ngrams(lemmatizing,1)
    #print("Unigrams ", Counter(unigram))
    bigram = ngrams(lemmatizing,2)
    #print("Bigrams ", Counter(bigram))
    
    fdist = FreqDist(lemmatizing)
    print("5 most common tokens ", fdist.most_common(5))
    
    fdist = FreqDist(unigram)
    print("5 most common unigrams", fdist.most_common(5))
    
    fdist = FreqDist(bigram)
    print("5 most common bigrams", fdist.most_common(5))
    
    return lemmatizing
    
'''
4.8 Write a program to compute unsmoothed unigrams and bigrams.
'''

def unsmoothedUnigramsBigrams(tokens):
    total = len(tokens)
    UnsmoothedUnigrams = Counter(ngrams(tokens, 1))
    UnsmoothedUnigramsProbabilies = {}
    
    for uni in UnsmoothedUnigrams:
        UnsmoothedUnigramsProbabilies[uni] = UnsmoothedUnigrams[uni]/total
    
    UnsmoothedBigrams = Counter(ngrams(tokens, 2))
    UnsmoothedBigramsProbabilies = {}
    
    for bigr in UnsmoothedBigrams:
        UnsmoothedBigramsProbabilies[bigr] = UnsmoothedBigrams[bigr]/total

    return UnsmoothedUnigramsProbabilies, UnsmoothedBigramsProbabilies

'''
4.9 Run your N-gram program on two different small corpora of your 
choice (you might use email text or newsgroups). Now compare the 
statistics of the two corpora. What are the differences in the most 
common unigrams between the two? How about interesting differences 
in bigrams?
'''
def compare_stats(path1, path2):
    # Reading  and processing data from the datasets
    data1 = load_data(path1)    
    processed_data1 = process_data(data1)
    
    data2 = load_data(path2)
    processed_data2 = process_data(data2)
    
    # Getting statistics for both datasets
    UnsmoothedUnigramsProbabilies1, UnsmoothedBigramsProbabilies1 = unsmoothedUnigramsBigrams(processed_data1)
    UnsmoothedUnigramsProbabilies2, UnsmoothedBigramsProbabilies2 = unsmoothedUnigramsBigrams(processed_data2)

    resU1 = {}
    for elem in UnsmoothedUnigramsProbabilies1:
        resU1[elem] = UnsmoothedUnigramsProbabilies1[elem]
    
    resU1 = OrderedDict(resU1)
    
    resU2 = {}
    for elem in UnsmoothedUnigramsProbabilies2:
        resU2[elem] = UnsmoothedUnigramsProbabilies2[elem]
    
    resU2 = OrderedDict(resU2)    
    
    resB1 = {}
    for elem in UnsmoothedBigramsProbabilies1:
        resB1[elem] = UnsmoothedBigramsProbabilies1[elem]
    
    resB1 = OrderedDict(resB1)
    
    resB2 = {}
    for elem in UnsmoothedBigramsProbabilies2:
        resB2[elem] = UnsmoothedBigramsProbabilies2[elem]
    
    resB2 = OrderedDict(resB2)  
    #print("resU1 ", len(resU1), " resU2 ", len(resU2), " resB1 ", len(resB1), " resB2 ", len(resB2))
    return resU1, resU2, resB1, resB2 
    

'''
4.10 Add an option to your program to generate random sentences.
'''
def generate_model(cfd, word, num=10):
    for i in range(num):
        print(word, end=' ')
        word = cfd[word].max()
        
'''    
4.11 Add an option to your program to compute the perplexity of a test set.
'''
def perplexity(tstbigram, trbigram, biprob):
    perplexity = 1
    tmp = 0
    for bigram in tstbigram:
        if bigram in trbigram and biprob[bigram] != 0:
            tmp += 1
            perplexity = perplexity * (1 / biprob[bigram])
    
    result = perplexity**(tmp)
    return result

def main():
    path = "/Users/Karina/Dataset/c_nokia.txt"
    tokens = load_data(path)
    processed_tokens = process_data(tokens)
    UnsmoothedUnigramsProbabilies, UnsmoothedBigramsProbabilies = unsmoothedUnigramsBigrams(processed_tokens)
    path1 = "/Users/Karina/Dataset/c_mp3.txt"
    path2 = "/Users/Karina/Dataset/c_router.txt"
    resU1, resU2, resB1, resB2 = compare_stats(path1, path2)
    bigrams = nltk.bigrams(processed_tokens)
    cfd = nltk.ConditionalFreqDist(bigrams)
    print(cfd['work'])
    print("Generated sentence> ")
    generate_model(cfd, 'work')
    
    tmp = len(processed_tokens)//3
    test_tokens = processed_tokens[:tmp]
    train_tokens = processed_tokens[tmp:]
    UnsmoothedUnigramsProbabiliesTr, UnsmoothedBigramsProbabiliesTr = unsmoothedUnigramsBigrams(train_tokens)
    UnsmoothedUnigramsProbabiliesTst, UnsmoothedBigramsProbabiliesTst = unsmoothedUnigramsBigrams(test_tokens)  
    bigrmtr = nltk.bigrams(train_tokens)
    bigrmtst = nltk.bigrams(test_tokens)
    print("Perplexity score> ",perplexity(bigrmtst,bigrmtr, UnsmoothedBigramsProbabiliesTst))
    
if __name__ == "__main__":
    main()
    