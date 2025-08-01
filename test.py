# A simple script to get the prediciton from the NLP model, by inputing the content of an email in the console

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text


# Load the BERT model (with KerasLayer for TF Hub)
model = tf.keras.models.load_model('D:\\Sheridan Cyber Security Program\\Year 4\\Semester 8\\ISS Graduation Project (Phase 2)\\NLP Model Code\\nlp_phishing_model', custom_objects={'KerasLayer':hub.KerasLayer})

# Prompt for input
print("Enter a test email body to test the NLP model:\n")
email = input()


print(f"Prediction: {model.predict([email])}")