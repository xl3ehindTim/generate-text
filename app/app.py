import os
import logging
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify
from accelerate import Accelerator

# Initialize Accelerator with automatic device mapping
accelerator = Accelerator(fp16=True)  # Enable mixed-precision for memory efficiency

# Load model and tokenizer (accelerator will handle device placement)
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Wrap model with accelerator for multi-GPU support
model, tokenizer = accelerator.prepare(model, tokenizer)

# Initialize Flask app
app = Flask(__name__)
  
@app.route("/generate", methods=["POST"])
def generate_text():
  # Get prompt and check for validity
  prompt = request.json.get("prompt")
  if not prompt:
    return "Error: Please provide a prompt in the 'prompt' field of the request body.", 400

  # Prepare inputs within accelerator context
  with accelerator.context():
    inputs = tokenizer(prompt, return_tensors="pt")
    model_inputs = accelerator.prepare_inputs_for_model(inputs)

    # Generate text
    output = model.generate(**model_inputs)
    decoded_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

  return jsonify({"generated_text": decoded_text})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5005)