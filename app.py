from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests

app = Flask(__name__)
CORS(app)

# Load Hugging Face API Key
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

@app.route("/api/recommend", methods=["POST"])
def recommend():
    data = request.json
    query = data.get("query")

    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    response = requests.post(
        "https://api-inference.huggingface.co/models/facebook/bart-large-cnn",
        headers=headers,
        json={"inputs": query},
    )

    return jsonify(response.json())

if __name__ == "__main__":
    app.run(debug=True)
