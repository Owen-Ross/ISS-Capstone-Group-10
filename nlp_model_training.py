
# Importing the required libraries for execution libraries
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Declaring the constant variables
num_epochs = 10
max_len = 200
num_words = 5000
oov_token = "<OOV>"

# Opening the training data file
df = pd.read_csv('D:\\Sheridan Cyber Security Program\\Year 4\\Semester 8\\ISS Graduation Project (Phase 2)\\NLP Model\\datasets\\CEAS_08.csv')
df.head()

# Initilizing tokenizer to convert words to numerical values
tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
tokenizer.fit_on_texts(df['body'])

# Saving the tokenizer to use in different scripts
with open('nlp_tokenizer.pickle', 'wb') as nlp_tokenizer:
    pickle.dump(tokenizer, nlp_tokenizer, protocol=pickle.HIGHEST_PROTOCOL)

# Creating a sequence of tokens, representing each body of the malicious email
sequences = tokenizer.texts_to_sequences(df['body'])

# Apply padding to all of the data, to ensure it is the same length 
padded = pad_sequences(sequences, maxlen = max_len, padding = 'post')

# Convert labels to numerical format (assuming binary classification: Phishing (1) or Legitimate (0))
labels = df['label'].values

# Splitting the data into test data and training data
X_train, X_test, y_train, y_test = train_test_split(padded, labels, test_size=0.2, random_state=42)

# Initilizing the neural network
model = tf.keras.Sequential([
    # This function will turn positive numbers into fixed length vectors
    tf.keras.layers.Embedding(num_words, 64, input_length=max_len),
    # Pooling the sequences into vectors of the same size
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Configuring the model for training
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model on the training data provided
model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test), verbose=2, batch_size=32)

# Saving ythe model for further trainning and testing
model.save('nlp_phishing_model.keras')