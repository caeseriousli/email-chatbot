import tensorflow as tf
import numpy as np
import pickle
import os

## IMPORTANT: please double check to make sure your current working directory is this project folder
## which includes the trained keras object as well as the tokenizer
cwd = os.getcwd()
model = tf.keras.models.load_model(os.path.join(cwd, 'data/tf_model_fit_10k_epoch10.keras'))
##
with open(os.path.join(cwd, 'data/tf_tokenizer_10k_epoch10'), 'rb') as loadfile:
    # Step 3
    tokenizer = pickle.load(loadfile)
##

# Start chatbot
num_pred_words = ''
question = ''

while not num_pred_words.isdigit():
    print('Please tell me how many words you\'d like to predict. Must be an integer.')
    num_pred_words = input('How many: ')
else:
    num_pred_words = int(num_pred_words)
    print('Please type at least 4 words to begin. Type quit to exit the program.')
    while 'quit' not in question:
        question = input('You: ')
        new_text = question.strip()
        for i in range(num_pred_words):
            temp_text = " ".join(new_text.split(" ")[-4:])
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