from datetime import datetime
 
import json
 
from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit
import collections
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import sklearn
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

print("Make model")
current = datetime.now()
model = Sequential()
model.add(Embedding(40000, 150, input_length=500))
model.add(Dropout(0.2))
model.add(Conv1D(75, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(150))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(datetime.now() - current)

print("Train model")
t1 = datetime.now()
model.fit(data, np.array(balanced_labels), validation_split=0.5, epochs=3)
print(datetime.now() - t1)

#In line two, we add an Embedding layer. This layer lets the network expand 
#each token to a larger vector, allowing the network to represent words in a 
#meaningful way. We pass 40000 as the first argument, which is the size of our 
#vocabulary (remember, we told the tokenizer to only use the 40000 most common 
#words earlier), and 150 as the second, which means that each token can be 
#expanded to a vector of size 150. We give it an input_length of 500, which is 
#the length of each of our sequences.

with open('/Users/Karina/Desktop/yelp/dataset/Grocery_and_Gourmet_Food.json', 'rb') as f:
    grocery = f.readlines()

gg = list(map(lambda x: x.rstrip(), grocery))
g = [json.loads(review) for review in gg]
gtexts = [tmp['reviewText'] for tmp in g]

binary_ratings = [0 if tmp['overall'] <=3 else 1 for tmp in g]
review_texts = []
true_labels = []
limit = len(gg)
count = [0, 0]
for i in range(len(gtexts)):
    polarity = binary_ratings[i]
    if count[polarity] < limit:
        review_texts.append(gtexts[i])
        true_labels.append(binary_ratings[i])
        count[polarity]+=1

test_data = gtexts
sequences = tokenizer.texts_to_sequences(test_data)
data = pad_sequences(sequences, maxlen=500)
predictions = model.predict(data)

'''
if prediction score <= 0.5 => assign sample to class 0
if prediction score > 0.5 => assign sample to class 1
'''
predictions[predictions>0.5]=1
predictions[predictions<=0.5]=0
print("Obtained prediction score", \
      sklearn.metrics.accuracy_score(true_labels, predictions))

'''
with open("keras_tokenizercnn.pickle", "wb") as f:
   pickle.dump(tokenizer, f)
model.save("yelp_sentiment_modelcnn.hdf5")

with open("keras_tokenizercnn.pickle", "rb") as f:
   tokenizer = pickle.load(f)

model = load_model("yelp_sentiment_modelcnn.hdf5")

# replace with the data you want to classify
newtexts = gtexts

sequences = tokenizer.texts_to_sequences(newtexts)
data = pad_sequences(sequences, maxlen=500)

predictions = model.predict(data)
print(predictions)
'''