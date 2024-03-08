import os
import logging
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify

# Load the model and tokenizer with GPU support (if available)
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
device = "cuda" if torch.cuda.is_available() else "cpu"  # Detect and use GPU if available
model = AutoModelForCausalLM.from_pretrained(model_name, device=device) # parameter: torch_dtype=torch.float16
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(torch.cuda.is_available())

# Initialize Flask application
app = Flask(__name__)


@app.route("/generate", methods=["POST"])
def generate_text():
  # Get the prompt from the request data
  prompt = request.json.get("prompt")

  # Check if prompt is provided
  if not prompt:
    return "Error: Please provide a prompt in the 'prompt' field of the request body.", 400

    # Prepare the input
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate text with custom maximum length
    max_length = 250
    output = model.generate(input_ids, max_length=max_length)

    # Decode the generated sequence
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Return the generated text
    return jsonify({"generated_text": generated_text})


if __name__ == "__main__":
  app.run(debug=True, host="0.0.0.0", port=5005)