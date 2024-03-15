import os
import logging
import torch
import json

from transformers import pipeline
from flask import Flask, request, jsonify

MODEL_NAME  =  "mistralai/Mistral-7B-Instruct-v0.2"
MAX_LENGTH  =  400
INSTRUCTION =  "My goal is to understand what is being talked about in a conversion. I want to know what the subjects of the conversation are and get some keywords from the conversation. Format the data as follows: [{subject: "", keywords: "" }, {subject: "", keywords: "" }] The conversation is as follows: "
GENERATE_KWARGS = {
  "do_sample": True,
  "temperature": 0.7,
  "max_new_tokens": MAX_LENGTH,
}

def build_prompt(instruction, conversation):
  return instruction + conversation


def extract_json(s):
  s = s[next(idx, c in enumerate(s) if c in "{["):]
  try:
    return json.loads(s)
  except json.JSONDecodeError as e:
    return json.loads(s[:e.pos])


# Preload model for efficiency on first run
generator = pipeline("text-generation", model=MODEL_NAME, device_map="auto")

app = Flask(__name__)


@app.route("/subjects", methods=["POST"])
def get_subjects():
  conversation = request.json.get("conversation")
  if not conversation:
    return "Error: No conversation parameter supplied in the request body", 400

  prompt = build_prompt(INSTRUCTION, conversation)

  output = generator(prompt, generate_kwargs=GENERATE_KWARGS)

  # Remove the input prompt from the output
  generated_text = output[0]["generated_text"].split(prompt)[1].strip()

  # Parse JSON
  parsed_output = extract_json(output[0]["generated_text"])

  return jsonify(parsed_output)


if __name__ == "__main__":
  app.run(debug=True, host="0.0.0.0", port=5005)