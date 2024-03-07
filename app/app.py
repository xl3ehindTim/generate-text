import os
import logging

from flask import Flask, request, jsonify
from transformers import pipeline

# Load the model pipeline
generator = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")

# Initialize Flask application
app = Flask(__name__)


@app.route("/generate", methods=["POST"])
def generate_text():
  # Get the prompt from the request data
  prompt = request.json.get("prompt")

  # Check if prompt is provided
  if not prompt:
    return "Error: Please provide a prompt in the 'prompt' field of the request body.", 400

  # Generate text using the model
  response = generator(prompt)

  # Return the generated text
  print(response)
  return jsonify(response[0]["generated_text"])


if __name__ == "__main__":
  app.run(debug=True, host="0.0.0.0", port=5005)