import os
import glob
import shutil
import random

## IMPORTANT: this file is not meant to be executed all at once.
# It contains the entire data wrangling and training pipeline
# Please read the code before adapting. Paths needed to be updated.

cwd = os.getcwd()

## IMPORTANT: find path to the enron dataset
## Finding all files ending with a '.' (all text files are named '1.', '2.', '3.'...)
data_path = '/Users/caesar/Downloads/trainingData/'
all_fnames = [y for x in os.walk(data_path) for y in glob.glob(os.path.join(x[0], '*.'))]

## Reading in individual files and writing to a single text file
### Select only 10k emails (og enron is too large)
fnames_10k = random.choices(all_fnames, k=200)
with open('/Users/caesar/Dropbox (Personal)/Personal/Career/ML/cleanData/enron_emails_200.txt','wb') as wfd:
    for f in fnames_10k:
        with open(f,'rb') as fd:
            shutil.copyfileobj(fd, wfd)


#############################################

### Parsing the emails to keep only the body #####
# Load data
with open('/Users/caesar/Dropbox (Personal)/Personal/Career/ML/cleanData/enron_emails_10k.txt', 'r', encoding='utf-8') as f:
    raw_data = f.read()

def parse_raw_message(raw_message):
    lines = raw_message.split('\n')
    # email = {}
    message = ''
    # keys_to_extract = ['from', 'to']
    for line in lines:
        if ':' not in line and line != '':
            # line = " ".join(w for w in nltk.wordpunct_tokenize(line) if w.lower() in words or not w.isalpha())
            message += line.strip() + '\n' ## remove leading and trailing spaces
    return message

raw_data = parse_raw_message(raw_data)


# Preprocess data
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

words = set(nltk.corpus.words.words())
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def preprocess(data, words, stop_words, lemmatizer):
    # Tokenize data
    tokens = nltk.word_tokenize(data)  # tokenizing needs punkt

    # Lowercase all words
    tokens = [word.lower() for word in tokens]

    # remove non-words according to nltk English library
    tokens = [word for word in tokens if word in words]

    # Remove stopwords and punctuation
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]

    # Lemmatize words
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens

# Preprocess data
processed_data = [preprocess(qa, words, stop_words, lemmatizer) for qa in raw_data.split('\n')]
processed_data = [i for i in processed_data if i != []]

###### Tensorflow steps, tokenizer, to sequence, and convert to 5-word training sets. ##
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.utils import to_categorical
import numpy as np

#####
num_train_words = 4  # how many trailing words do we use to train to predict the next word?
data = processed_data
# vocab_size = 5000
tokenizer = Tokenizer(oov_token = '<OOV>') # add out-of-vocabulary token
tokenizer.fit_on_texts(data)
# determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# create line-based sequences
sequences = list()
for encoded in tokenizer.texts_to_sequences(data):
    if len(encoded) > 0:
        for i in range(0, len(encoded) - num_train_words):
            sequences.append(encoded[i:i + num_train_words + 1])

print('Total Sequences: %d' % len(sequences))
sequences = np.array(sequences)
X, y = sequences[:, :-1], to_categorical(sequences[:, -1], num_classes=vocab_size)

# define model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1])),
    tf.keras.layers.Embedding(vocab_size, output_dim=10),
    tf.keras.layers.LSTM(10),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])
print(model.summary())
# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# tf.keras.utils.plot_model(model)


model.fit(X, y, epochs=10, verbose=2)

## Save the object
import pickle

model.save('/Users/caesar/Dropbox (Personal)/Personal/Career/ML/chatbot1/tf_model_fit_10k_epoch10.keras')
with open('/Users/caesar/Dropbox (Personal)/Personal/Career/ML/chatbot1/tf_tokenizer_10k_epoch10', 'wb') as savefile:
    # Step 3
    pickle.dump(tokenizer, savefile)


model = keras.models.load_model('/Users/caesar/Dropbox (Personal)/Personal/Career/ML/chatbot1/test.keras')
##
# with open('/Users/caesar/Dropbox (Personal)/Personal/Career/ML/chatbot1/tf_model_fit_10k', 'rb') as loadfile:
#     # Step 3
#     model = pickle.load(loadfile)
# with open('/Users/caesar/Dropbox (Personal)/Personal/Career/ML/chatbot1/tf_tokenizer_10k', 'rb') as loadfile:
#     # Step 3
#     tokenizer = pickle.load(loadfile)
##
### Prediction ######
num_pred_words = 10
text = "Could you please send me"

new_text = text.strip()
for i in range(num_pred_words):
    temp_text = " ".join(new_text.split(" ")[-num_train_words:])
    # Preprocessing
    # temp_text = preprocess(temp_text)
    #
    encoded = tokenizer.texts_to_sequences([temp_text])[0]
    encoded = np.array([encoded])
    next = model.predict(encoded, verbose=0)
    for x in next:
        next_word_token = np.argmax(x)
        # map predicted word index to word
        for word, index in tokenizer.word_index.items():
            if index == next_word_token:
                new_text += " " + word
print(new_text)

# Start chatbot
num_pred_words = 10
question = ''
print('Please type 3 words to begin. Type quit to exit the program')
while 'quit' not in question:
    question = input('You: ')
    new_text = question.strip()
    for i in range(num_pred_words):
        temp_text = " ".join(new_text.split(" ")[-3:])
        encoded = tokenizer.texts_to_sequences([temp_text])[0]
        encoded = np.array([encoded])
        next = model.predict(encoded, verbose=0)
        for x in next:
            next_word_token = np.argmax(x)
            # map predicted word index to word
            for word, index in tokenizer.word_index.items():
                if index == next_word_token:
                    new_text += " " + word
    print('Chatbot:', new_text)


