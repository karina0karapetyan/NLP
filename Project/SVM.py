from datetime import datetime
import json
from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import timeit
import collections
import pickle
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
import sklearn
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

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

print("Vectorizer")
t1 = datetime.now()
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=3)
classifier = LinearSVC()
out = vectorizer.fit_transform(balanced_texts)

print("Time")
print("shape", out.shape)

score = cross_val_score(classifier, out, balanced_labels, cv=5, n_jobs=-1)
print("Time")
print("Score> ")
print(score)
print("sum(score) / len(score)")
print(sum(score) / len(score))

model = classifier.fit(out, balanced_labels)

with open('/Users/Karina/Desktop/yelp/dataset/Grocery_and_Gourmet_Food.json', 'rb') as f:
    grocery = f.readlines()

gg = list(map(lambda x: x.rstrip(), grocery))
g = [json.loads(review) for review in gg]
gtexts = [tmp['reviewText'] for tmp in g]

# Convert Grocery_and_Gourmet_Food ratings into (negative or positive)
binary_ratings = [0 if tmp['overall'] <= 3 else 1 for tmp in g]
review_texts = []
true_labels = []
limit = len(gg)  
count = [0, 0]
for i in range(len(gtexts)):
    polarity = binary_ratings[i]
    if count[polarity] < limit:
        review_texts.append(gtexts[i])
        true_labels.append(binary_ratings[i])
        count[polarity] += 1

test_data = gtexts

outt = vectorizer.transform(test_data)

predictions = model.predict(outt)
print("Obtained prediction score", \
      sklearn.metrics.accuracy_score(true_labels, predictions))
print(datetime.now() - t1)


# Compute confusion matrix
cnf_matrix = confusion_matrix(true_labels, predictions)
class_names = ["Negative Reviews", "Positive Reviews"]
# Plot non-normalized confusion matrix
plt.figure()
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Confusion matrix for the SVM classifier')

plt.show()

# 120044 - positive feedback
# 31210 - negative feedback
