"""
Filename: test.py
Created by: Owen Ross <rossow@sheridancollege.ca>
Created on: July 11, 2025

Last modeified by: Owen Ross <rossow@sheridancollege.ca>
Last modeified on: August 7, 2025

Description: A simple script to test the model, by getting input from the user, and displaying
the prediction that the model has given
"""

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text


# Load the BERT model (with KerasLayer for TF Hub)
# Change below file path to where the pre-trained model is on your system
model = tf.keras.models.load_model('D:\\Sheridan Cyber Security Program\\Year 4\\Semester 8\\ISS Graduation Project (Phase 2)\\NLP Model Code\\nlp_phishing_model', custom_objects={'KerasLayer':hub.KerasLayer})

# Prompt for input
print("Enter a test email body to test the NLP model:\n")
email = input()


# Print the model's prediction
print(f"Prediction: {model.predict([email])}")