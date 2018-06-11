'''
Implement Logistic Regression with PyTorch for the dataset
'''
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
import nltk
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import string 
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.util import ngrams
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch import optim
from sklearn.metrics import accuracy_score

stemmer = SnowballStemmer("english")
tokenizer = nltk.RegexpTokenizer(r"\w+")

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

    return stemming

# process labels/features
def process(path):
    label_feats = []  
    with open(path, mode='r') as f:
        for line in f.readlines():
            if "$ neg $" in line:
                labels = 0
                tmp = [x.strip() for x in line.split("$ neg $")]
                temp = process_data(tmp[1])
            if "$ pos $" in line:
                labels = 1
                tmp = [x.strip() for x in line.split("$ pos $")]
                temp = process_data(tmp[1])
            
            label_feats.append((labels, tmp[1]))
    
    return label_feats
 
# split the dataset into 75% training and 25% testing sets
def train_test(data):
    shuffle(data)
    trainingset = data[:int(len(data) * 0.75)]
    testset = data[int(len(data) * 0.75):]
    return trainingset, testset

class Data(Dataset):
    def __init__(self, data_tensor, target_tensor):
        assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
    
    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]
    
    def __len__(self):
        return self.data_tensor.size(0)

class Linear_model(nn.Module):
    
    def __init__(self, VOCABULARYab_size, num_labels):
        super(Linear_model, self).__init__()
        self.linear = nn.Linear(VOCABULARYab_size, num_labels)
    
    def forward(self, text_vector):
        return self.linear(text_vector)
    
def main():
    path = "/Users/Karina/Desktop/NLPCourse/NLP/feature-specific/Dataset2.txt"
    label_feats = process(path)
    trainingset, testset = train_test(label_feats)
    trX = [doc[1] for doc in trainingset]
    trY = [doc[0] for doc in trainingset]
    teX = [doc[1] for doc in testset]
    teY = [doc[0] for doc in testset]
    VOCABULARY = set(' '.join(trX).split())
    dimension = len(VOCABULARY)
    word_to_ix = {}
    for _, tokens in label_feats:
        for word in tokens:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    
    trainTensor = np.zeros((len(trX), dimension))
    for i,desc in enumerate(trX):
        tmp = np.zeros(dimension)
        for w in desc.lower().split():
            if w in word_to_ix:
                tmp[word_to_ix[w]] = 1
        trainTensor[i] = tmp

    trainTensor = torch.FloatTensor(trainTensor)
    trainlblTensor = torch.LongTensor(np.array(trY))
    trainftrTensor = np.zeros((len(teX), dimension))
    for i,desc in enumerate(teX):
        tmp = np.zeros(dimension)
        for w in desc.lower().split():
            if w in word_to_ix:
                tmp[word_to_ix[w]] = 1
        trainftrTensor[i] = tmp
    
    trainftrTensor = torch.FloatTensor(trainftrTensor)
    testlblTensor = torch.LongTensor(np.array(teY))
   
    train_loader = DataLoader(Data(trainTensor, trainlblTensor), batch_size=1)
    test_loader = DataLoader(Data(trainftrTensor, testlblTensor), batch_size=1)
    model = Linear_model(dimension, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    for epoch in range(0,11):
        for desc, label in train_loader:
            desc = Variable(desc)
            label = Variable(label)
            optimizer.zero_grad()
            out = model(desc)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            
    predictions = []
    for desc, label in test_loader:
        desc = Variable(desc)
        out = model(desc).view(-1)
        topval, topindex = out.data.max(0)
        predictions.append(topindex[0])
    print(accuracy_score(predictions, teY))
    
    
if __name__ == "__main__":
    main() 
    
    
    