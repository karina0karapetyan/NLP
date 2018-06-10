from datetime import datetime
 
import json
 
from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import timeit
import collections

'''
read and process reviews
review.json is available at:
https://www.yelp.com/dataset/documentation/json
'''

t1 = datetime.now()
# read the entire file into a python array
with open('/Users/Karina/Desktop/yelp/dataset/review.json', 'rb') as f:
    reviews = f.readlines()

# remove the trailing "\n" from each line
reviews = list(map(lambda x: x.rstrip(), reviews))
reviews = [json.loads(review) for review in reviews]
print(datetime.now() - t1)

df = pd.DataFrame(reviews)
df.describe().transpose()

np.random.seed(10)
plt.figure(figsize=(8, 5))
plt.hist(df['stars'], alpha=0.6, label="Stars")
plt.legend(shadow=True, loc=(0.04, 0.86), fontsize=12)
plt.title("Class Distribution of the Yelp dataset", fontweight='bold', color= 'b', fontsize=16)
plt.xlabel('Amount of Stars regarding the reviews', {'color': 'b', 'fontsize': 14})
plt.ylabel('Frequency', {'color': 'b', 'fontsize': 14})
plt.grid()
plt.savefig('/Users/Karina/Desktop/yelp/histogram.png')

# Get a balanced sample of positive and negative reviews
texts = [review['text'] for review in reviews]

# Convert our 5 classes into 2 (negative or positive)
binstars = [0 if review['stars'] <= 3 else 1 for review in reviews]
balanced_texts = []
balanced_labels = []
limit = 200000  # Change this to grow/shrink the dataset
neg_pos_counts = [0, 0]
for i in range(len(texts)):
    polarity = binstars[i]
    if neg_pos_counts[polarity] < limit:
        balanced_texts.append(texts[i])
        balanced_labels.append(binstars[i])
        neg_pos_counts[polarity] += 1
        
print(collections.Counter(balanced_labels))
# >>> Counter({1: 200000, 0: 200000})

# 1/5 of limit variable
tokenizer = Tokenizer(num_words=40000)
tokenizer.fit_on_texts(balanced_texts)
sequences = tokenizer.texts_to_sequences(balanced_texts)
data = pad_sequences(sequences, maxlen=500)

print("Carrying out LSTM model")
current = datetime.now()
model = Sequential()
model.add(Embedding(40000, 150, input_length=500))
model.add(LSTM(150, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', \
              metrics=['accuracy']) 
print("Finished establishing LSTM", datetime.now() - current)

#In line two, we add an Embedding layer. This layer lets the network expand 
#each token to a larger vector, allowing the network to represent words in a 
#meaningful way. We pass 40000 as the first argument, which is the size of our 
#vocabulary (remember, we told the tokenizer to only use the 40000 most common 
#words earlier), and 150 as the second, which means that each token can be 
#expanded to a vector of size 150. We give it an input_length of 500, which is 
#the length of each of our sequences.
print("Train model")
t1 = datetime.now()
model.fit(data, np.array(balanced_labels), validation_split=0.5, epochs=3)
print(datetime.now() - t1)

# save the tokenizer and model
with open("/Users/Karina/Desktop/yelp/LSTM/keras_tokenizer.pickle", "wb") as f:
   pickle.dump(tokenizer, f)

model.save("yelp_sentiment_model.hdf5")


# load the tokenizer and the model
with open("/Users/Karina/Desktop/yelp/LSTM/keras_tokenizer.pickle", "rb") as f:
   tokenizer = pickle.load(f)

model = load_model("/Users/Karina/Desktop/yelp/LSTM/yelp_sentiment_model.hdf5")

# replace with the data you want to classify
newtexts = ["One of the most remarkable experiences in my life!", "I will never visit this place again!",]

# note that we shouldn't call "fit" on the tokenizer again
sequences = tokenizer.texts_to_sequences(newtexts)
data = pad_sequences(sequences, maxlen=500)

# get predictions for each of your new texts
predictions = model.predict(data)
print(predictions)
