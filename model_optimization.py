import sys
import csv
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import keras_tuner as kt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Increasing the max limit for reading CSV fields
csv.field_size_limit(100000000)

# Opening the training data file    
# df = pd.read_csv("D:\\Sheridan Cyber Security Program\\Year 4\\Semester 8\\ISS Graduation Project (Phase 2)\\Training Data\\SpamAssasin.csv")
df = pd.read_csv("D:\\Sheridan Cyber Security Program\\Year 4\\Semester 8\\ISS Graduation Project (Phase 2)\\Training Data\\CEAS_08.csv")

# Drop any rows with empty body or label
df.dropna(subset=["body"], inplace=True)
df.dropna(subset=["label"], inplace=True)

# # Splitting the data into test data and training data
x_train, x_test, y_train, y_test = train_test_split(df["body"], df["label"].values, stratify=df["label"], test_size=0.25, random_state=42)

def model_builder(hp):
    # Import the uncased bert model for preprocessing text
    preprocessor = hub.KerasLayer(
        "https://kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-preprocess/3")

    # Import the uncased bert model for encoding the text into sequences
    encoder = hub.KerasLayer(
        "https://kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-l-12-h-768-a-12/3",
        trainable=True)
    
    text_input  = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text_input")
    preprocessed_text = preprocessor(text_input)
    outputs = encoder(preprocessed_text)

    hp_units = hp.Int('units', min_value=32, max_value=256, step=32)

    layer = tf.keras.layers.Dense(hp_units, activation='relu')(outputs["pooled_output"])
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

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-4, 3e-4, 1e-3])

    # Setting the configuration for the model training
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss="binary_crossentropy",
                metrics=METRICS)
    
    return model


tuner = kt.Hyperband(model_builder,
                    objective=kt.Objective("val_accuracy", direction="max"),
                    max_epochs=10,
                    factor=3,
                    directory='my_dir',
                    project_name='phishing_detection')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(x_train, y_train, epochs=50, validation_data=(x_test, y_test), callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")