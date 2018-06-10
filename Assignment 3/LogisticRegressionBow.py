'''
    Implement Logistic Regression where the input BOW of the input using 
    the same sentiment dataset. Compare the results with the simple 
    Logistic Regression
    Reference: https://pytorch.org/tutorials/beginner/nlp/deep_learning_tutorial.html
'''

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

# split the dataset into 75% training and 25% testing sets
def train_test(data):
    shuffle(data)
    trainingset = data[:int(len(data) * 0.75)]
    testset = data[int(len(data) * 0.75):]
    return trainingset, testset

class BoWClassifier(nn.Module):  # inheriting from nn.Module!

    def __init__(self, vocab_size):
        # calls the init function of nn.Module.  Dont get confused by syntax,
        # just always do it in an nn.Module
        super(BoWClassifier, self).__init__()

        # Define the parameters that you will need.  In this case, we need A and b,
        # the parameters of the affine mapping.
        # Torch defines nn.Linear(), which provides the affine map.
        # Make sure you understand why the input dimension is vocab_size
        # and the output is num_labels!
        self.linear = nn.Linear(vocab_size, 1)

        # NOTE! The non-linearity log softmax does not have parameters! So we don't need
        # to worry about that here

    def forward(self, bow_vec):
        # Pass the input through the linear layer,
        # then pass that through log_softmax.
        # Many non-linearities and other functions are in torch.nn.functional
        return F.sigmoid(self.linear(bow_vec))

def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        vec[word_to_ix[word]] += 1
    return vec.view(1, -1)

def make_target(label, label_to_ix):
    return torch.Tensor([label_to_ix[label]])

# train model
def train(model, trainingset, word_to_ix):
    label_text_int = {"neg": 0, "pos": 1}
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    temp_loss = []
    for epoch in range(51):  
        print("epoch id> ", epoch) 
        epoch_loss = 0
        for label, tokens in trainingset:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
            # Step 2. Make our BOW vector and also we must wrap the target in a
            # Variable as an integer. For example, if the target is SPANISH, then
            # we wrap the integer 0. The loss function then knows that the 0th
            # element of the log probabilities is the log probability
            # corresponding to SPANISH
            bow_vec = autograd.Variable(make_bow_vector(tokens, word_to_ix))
            target = autograd.Variable(make_target(label, label_text_int))
            # Step 3. Run our forward pass.
            pred = model(bow_vec)
            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss = loss_function(pred[0], target)
            epoch_loss += loss.data
            loss.backward()
            optimizer.step()
        
        temp_loss.append(epoch_loss)
      
def get_accuracy(model, testset, word_to_ix):
    correct = 0
    for label, tokens in testset:
        pred = model(autograd.Variable(make_bow_vector(tokens, word_to_ix)))
        if pred[0] > 0.5:
            if label == "pos":
                correct += 1
        else:
            if label == "neg":
                correct += 1
    print("Accuracy score> ", correct/len(testset))

def main():
    path = "/Users/Karina/Desktop/NLPCourse/NLP/feature-specific/Dataset2.txt"
    label_feats = process(path)
    trainingset, testset = train_test(label_feats)
    
    # word_to_ix maps each word in the vocab to a unique integer, which will be its
    # index into the Bag of words vector
    word_to_ix = {}
    for _, tokens in label_feats:
        for word in tokens:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
            
    VOCAB_SIZE = len(word_to_ix)
    model = BoWClassifier(VOCAB_SIZE)
    # the model knows its parameters.  The first output below is A, the second is b.
    # Whenever you assign a component to a class variable in the __init__ function
    # of a module, which was done with the line
    # self.linear = nn.Linear(...)
    # Then through some Python magic from the Pytorch devs, your module
    # (in this case, BoWClassifier) will store knowledge of the nn.Linear's parameters
    '''
    for param in model.parameters():
        print(param)
    '''
    # To run the model, pass in a BoW vector, but wrapped in an autograd.Variable
    sample = trainingset[0]
    bow_vector = make_bow_vector(sample[0], word_to_ix)
    pred = model(autograd.Variable(bow_vector)) 
    train(model, trainingset, word_to_ix)
    get_accuracy(model, testset, word_to_ix)
    
if __name__ == "__main__":
    main()
    