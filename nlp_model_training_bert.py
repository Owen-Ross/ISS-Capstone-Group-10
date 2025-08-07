"""
Filename: nlp_model_training_bert.py
Created by: Owen Ross <rossow@sheridancollege.ca>
Created on: July 9, 2025

Last modeified by: Owen Ross <rossow@sheridancollege.ca>
Last modeified on: August 7, 2025

Description: This script trains a binary classification model using BERT from TensorFlow Hub 
to detect phishing emails based on their body content. The trained model is saved for future 
inference in a browser extension or other applications.
"""
import sys
import csv
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Displaying the available GPUs
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available:", len(physical_devices))
if physical_devices:
    print("Using GPU:", physical_devices[0].name)
else:
    print("GPU not found. Using CPU instead.")

# Increasing the max limit for reading CSV fields
csv.field_size_limit(100000000)

# Opening the training data file    
df = pd.read_csv("D:\\Sheridan Cyber Security Program\\Year 4\\Semester 8\\ISS Graduation Project (Phase 2)\\Training Data\\CEAS_08.csv")

# Drop any rows with empty body or label
df.dropna(subset=["body"], inplace=True)
df.dropna(subset=["label"], inplace=True)

# # Splitting the data into test data and training data
x_train, x_test, y_train, y_test = train_test_split(df["body"], df["label"].values, stratify=df["label"], test_size=0.30, random_state=42)

# Printing the amount of training and testing data for debugging purposes
# Commented out below code for debugging
# print(f"Size of Training Data: {len(x_train)} - {len(y_train)}")
# print(f"Size of Test Data: {len(x_test)} - {len(y_test)}")

# Import the uncased bert model for preprocessing text
preprocessor = hub.KerasLayer(
    "https://kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-preprocess/3")

# Import the uncased bert model for encoding the text into sequences
encoder = hub.KerasLayer(
    "https://kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-l-12-h-768-a-12/3",
    trainable=True)

# Preprocessing text with BERT
text_input  = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text_input")
preprocessed_text = preprocessor(text_input)
outputs = encoder(preprocessed_text)

# Neural network layers
layer = tf.keras.layers.Dense(32, activation='relu')(outputs["pooled_output"])
output = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(layer)


# Constructing model
model = tf.keras.Model(inputs=[text_input], outputs=[output])

# The metrics that will be used to measure the performance of the model
METRICS = [
    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.FalsePositives(name="false positives"),
    tf.keras.metrics.AUC(name="auc")
]

# Setting the configuration for the model training
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss="binary_crossentropy",
              metrics=METRICS)

# Ensure proper data types before training
x_train = x_train.astype(str)
x_test = x_test.astype(str)
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Training the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=[
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)])

# Printing the model summary
print(f"\n{model.summary()}")

# Predict probabilities and convert to binary labels
y_pred_probs = model.predict(x_test)
y_pred = (y_pred_probs > 0.5).astype("int32")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Saving ythe model for further trainning and testing
model.save('nlp_phishing_model', save_format='tf')