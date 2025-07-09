import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import shap

# Load the BERT model (with KerasLayer for TF Hub)
model = tf.keras.models.load_model('nlp_phishing_model.keras', custom_objects={'KerasLayer':hub.KerasLayer})

# Define prediction function that accepts raw strings
def predict_fn(texts):
    return model(tf.constant(texts, dtype=tf.string))

# Use a small subset for explanation
sample_texts = ["You have been selected for a free trip to the Bahamas!"]

# Create a text masker explicitly
masker = shap.maskers.Text()

# Create the SHAP explainer for text input
explainer = shap.Explainer(predict_fn, masker)

# Get SHAP values
shap_values = explainer(sample_texts)

# Save SHAP text explanation to HTML
html_output = shap.plots.text(shap_values[0], display=False)

prediction = model(tf.constant(sample_texts, dtype=tf.string))[0][0]

# Inject prediction into HTML
prediction_html = f"""
    <div style='font-family: Arial; font-size: 18px; margin-bottom: 10px;'>
        <strong>Model Prediction:</strong> {prediction}
    </div>
"""
full_html = f"{prediction_html}\n{html_output}"

# Save to HTML file
with open("shap_text_output.html", "w", encoding="utf-8") as f:
    f.write(full_html)
