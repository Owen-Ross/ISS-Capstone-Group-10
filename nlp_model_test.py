"""
Filename: nlp_model_test.py
Created by: Owen Ross <rossow@sheridancollege.ca>
Created on: May 11, 2025

Last modeified by: Owen Ross <rossow@sheridancollege.ca>
Last modeified on: August 7, 2025

Description: Using the LIME library to explain the prediciton the model gave for the
input the user provides.

NOTE: This script is no longer being used, this is included to show the progress of the project.
"""

# Import the libraries that are needed for execution
import tensorflow as tf
import pickle
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from lime.lime_text import LimeTextExplainer

# Overriding the KerasTokenizerWrapper class, because LIME expects a transform function when analyzing a prediction for a model
class KerasTokenizerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, tokenizer, maxlen=100):
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def fit(self, X, y=None):
        return self

    # Transforms tokenized text into sequences to be analized by the NLP model
    def transform(self, X):
        sequences = self.tokenizer.texts_to_sequences(X)
        return tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=self.maxlen)

# Overriding the KerasModelWrapper class, becasue LIME calls these functions when it is trying to understand the model
class KerasModelWrapper(BaseEstimator):
    def __init__(self, model):
        self.model = model

    # since the model is already fitted the function will just return the already trained model
    def fit(self, X, y=None):
        return self

    # A function that will get the probability from the model
    def predict_proba(self, X):
        probs = self.model.predict(X)
        # Converts the one dimensional array, into a two dimensional array that LIME expects  
        if probs.shape[1] == 1:
            # Gets the probablity for both categories
            probs = np.hstack([1 - probs, probs])
        return probs


# Load the trained model
model = tf.keras.models.load_model('nlp_phishing_model.keras')

# Load tokenizer that was used to train data
with open('nlp_tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)


# Take the email text the user entered, and get an explaination on the prediction it gave for the text 
def explain_decision(email, pipe):
    # Set the names for the 2 options the model can make a prediction for
    explainer = LimeTextExplainer(class_names=["benign", "phishing"])
    # Get the explaination from LIME
    explanation = explainer.explain_instance(email, pipe.predict_proba, num_features=10)
    # Save the explaination from LIME and save it in an HTML file
    explanation.save_to_file('nlp_model_results.html')


# Wrapping the tokenizer and the NLP model to be used with LIME
tokenizer_wrapper = KerasTokenizerWrapper(tokenizer=tokenizer, maxlen=200)
wrapped_model = KerasModelWrapper(model)

pipe = Pipeline([
    ('tokenizer', tokenizer_wrapper),
    ('model', wrapped_model)
])

# Prompt for input
print("Enter a test email body to test the NLP model:\n")
email = input()

# Explain prediction
explain_decision(email, pipe)

# Print the tensorflow model prediction

