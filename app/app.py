import os
import logging
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify

# Load the model and tokenizer with GPU support (if available)
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
device = "cuda" if torch.cuda.is_available() else "cpu" 
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16) # check if float32 runs
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize Flask application
app = Flask(__name__)


@app.route("/extract", methods=["POST"])
def extract_topics(text):
  text = request.json.get("prompt")

  prompt = f"What are the main topics discussed in the following conversation, please return a JSON format and nothing more: {text}"
  inputs = tokenizer(prompt, return_tensors=device)
  output = model.generate(**inputs)
  decoded_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

  # Identify potential JSON start and end
  start_index = decoded_text.find('{')
  end_index = decoded_text.rfind('}') 

  # Extract potential JSON content (if valid indices found)
  if start_index >= 0 and end_index > start_index:
    json_content = decoded_text[start_index:end_index+1]
  else:
    json_content = None  # No valid JSON found

  # Try parsing and handling potential errors
  try:
    if json_content:
      data = json.loads(json_content)
      topics = [{"Topic": topic["Topic"]} for topic in data["Topics"]]  
      return jsonify.dumps({"Topics": topics})
    else:
      return json.dumps({"Error": "No valid JSON found"})
  except json.JSONDecodeError:
    return jsonify.dumps({"Error": "Invalid JSON format"})



@app.route("/generate", methods=["POST"])
def generate_text():
  # Get the prompt from the request data
  prompt = request.json.get("prompt")

  # Check if prompt is provided
  if not prompt:
    return "Error: Please provide a prompt in the 'prompt' field of the request body.", 400

  inputs = tokenizer(prompt, return_tensors=device)
  output = model.generate(**inputs)
  decoded_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

  # Prepare the input
  input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

  # Generate text with custom maximum length
  max_length = 500
  output = model.generate(input_ids, max_new_tokens)

  # Decode the generated sequence
  generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

  # Return the generated text
  return jsonify({"generated_text": generated_text})


if __name__ == "__main__":
  app.run(debug=True, host="0.0.0.0", port=5005)