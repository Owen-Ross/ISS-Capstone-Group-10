import sys
import csv
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available:", len(physical_devices))
if physical_devices:
    print("Using GPU:", physical_devices[0].name)
else:
    print("GPU not found. Using CPU instead.")

# Increasing the max limit for reading CSV fields
csv.field_size_limit(100000000)

# Opening the training data file    
# df = pd.read_csv("D:\\Sheridan Cyber Security Program\\Year 4\\Semester 8\\ISS Graduation Project (Phase 2)\\Training Data\\SpamAssasin.csv")
df = pd.read_csv("D:\\Sheridan Cyber Security Program\\Year 4\\Semester 8\\ISS Graduation Project (Phase 2)\\Training Data\\CEAS_08.csv")

# Combine sender, subject, and body into a single input string
df["combined_text"] = df["sender"].astype(str) + " [SEP] " + df["subject"].astype(str) + " [SEP] " + df["body"].astype(str)

df.dropna(subset=["combined_text", "label"], inplace=True)

# Drop any rows with empty body or label
# df.dropna(subset=["body"], inplace=True)
# df.dropna(subset=["label"], inplace=True)

# # Splitting the data into test data and training data
x_train, x_test, y_train, y_test = train_test_split(df["combined_text"], df["label"].values, stratify=df["label"], test_size=0.25, random_state=42)

print(f"Size of Training Data: {len(x_train)} - {len(y_train)}")
print(f"Size of Test Data: {len(x_test)} - {len(y_test)}")

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
# Commented out below layers for testing purposes
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

print(f"\n{model.summary()}")

# Predict probabilities and convert to binary labels
y_pred_probs = model.predict(x_test)
y_pred = (y_pred_probs > 0.5).astype("int32")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Saving ythe model for further trainning and testing
model.save('nlp_phishing_model', save_format='tf')