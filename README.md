## email-chatbot

### Introduction
A tiny toy example of an email chatbot trained on randomly selected 10k emails from enron dataset. It currently has just an input layer which takes in 4 words/tokens at a time, a small embedding layer, a recurrent LSTM layer, and a dense layer. At this stage the bot is not yet coherent, more like churning out related words. It is, however, surprisingly functional and quite cool to play with, considering its small training set, tiny net, and 10 epochs only. Will build upon this and training more sophisticated models.

TLDR: to use the chatbot, you can either execute in command line with `python3 email_chatbot.py` or run `email_chatbot.py` code in your favorite IDE. To see my pipeline of training this nn, see `/archive/archive_training_code.py`

**A screenshot of the chat box**

<img src="data/chat_screenshot.png?raw=true" width="700"/>

### Briefly the Method
The 10k randomly selected emails from the enron dataset were first combined and cleaned by parsing out lines containing ":" with "to" and "from". Then with the help of NLTK, the raw data were tokenized, filtered to have only English words/tokens, removed stop words and punctuations, and lemmatized. At this point, if one attempted my pipeline on the entirety of enron dataset, should get a very clean text data without all random symbols, email addresses, and random letters in the dataset. This pre-processed text data are then fed to TensorFlow keras Tokenizer and converted to sequence. The sequence is segmented into 5-token lists (using every 4 tokens to predict the next token, moving the window one token at a time), combined into a numpy array with np.shape=(N,5). X, y are the first 4 and the last column, respectively.  




| Layer (type)    |            Output Shape        |     Param    |
|:----------- |:----------- |:------------ |
| input_3 (InputLayer)    |   (None, 4)       |       0     |    
| embedding_2 (Embedding)  |   (None, 4, 10)     |    155410   |
| lstm_2 (LSTM)        |       (None, 10)          |      840  |      
| dense_2 (Dense)     |        (None, 15541)      |       170951   |

---

Total params: 327,201  
Trainable params: 327,201
Non-trainable params: 0

---



