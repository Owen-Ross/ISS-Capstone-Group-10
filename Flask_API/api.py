"""
Filename: api.py
Created by: Owen Ross <rossow@sheridancollege.ca>
Created on: August 1, 2025

Last modeified by: Owen Ross <rossow@sheridancollege.ca>
Last modeified on: August 7, 2025

Description: This API application, gets the credentials from Google, sends the email content to the
NLP model endpoint, and send the prediction from the NLP model to the browser extension.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import requests
import google.auth
from google.auth.transport.requests import Request

app = Flask(__name__)

# Allows the API to make cross-domain asyncronous requests in chrome
CORS(app, origins=["chrome-extension://mecgfbiblohefgmkojmgikbmjjcjklm"])

# Getting the required credentials from Google
credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])

# IDs and information required to connect to Google Vertex AI
PROJECT_ID = "578705582953"
REGION = "us-central1"
ENDPOINT_ID = "2147615589395333120"
ENDPOINT_URL = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}:predict"

# Endpoint information for API
@app.route('/predict', methods=['POST'])
def prediction():
    # Getting the JSON request from the browser extension, with the email content
    data = request.get_json()
    email_body = data.get("body")
    
    # Getting the authentication token using the Google credentials
    credentials.refresh(Request())
    token = credentials.token

    # Setting the headers and payload contents for the 
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {"instances": [email_body]}

    # Sending the request to the model endpoint, and getting the response
    response = requests.post(ENDPOINT_URL, headers=headers, json=payload)
    return jsonify(response.json()), response.status_code