import tensorflow as tf
import numpy as np
from lime.lime_text import LimeTextExplainer

# Load the BERT model (with KerasLayer for TF Hub)
model = tf.keras.models.load_model('nlp_phishing_model.keras', custom_objects={
    'KerasLayer': tf.keras.layers.Layer
})

# Define the predict_proba function for LIME
def predict_proba(texts):
    # Ensure input is a numpy array of strings
    texts = np.array(texts, dtype=str)
    probs = model.predict(texts)

    # Convert [N, 1] to [N, 2] for binary classification (benign or phishing)
    if probs.shape[1] == 1:
        probs = np.hstack([1 - probs, probs])
    return probs

# Function to explain model's decision
def explain_decision(email):
    explainer = LimeTextExplainer(class_names=["benign", "phishing"])
    explanation = explainer.explain_instance(email, predict_proba(email), num_features=10)
    explanation.save_to_file('nlp_model_results.html')

# Prompt for input
print("Enter a test email body to test the NLP model:\n")
email = input()

# Run explanation
explain_decision(email)
