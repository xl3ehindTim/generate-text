import json

from transformers import pipeline
from flask import Flask, request, jsonify
from .helpers import build_prompt, extract_json

MODEL_NAME  =  "mistralai/Mistral-7B-Instruct-v0.2"
INSTRUCTION =  "My goal is to understand what is being talked about in a conversion. I want to know what the subjects of the conversation are and get some keywords from the conversation. Format the data as follows: [{subject: "", keywords: "" }, {subject: "", keywords: "" }] The conversation is as follows: "


# Preload model for efficiency on first run
generator = pipeline("text-generation", model=MODEL_NAME, device_map="auto")

app = Flask(__name__)


@app.route("/subjects", methods=["POST"])
def get_subjects() -> flask.Response:
  """
  This function analyzes a conversation text and identifies potential subjects with 
  associated keywords using a large language model.

  **Expects a POST request with the following JSON data in the request body:**

  ```json
  {
    "conversation": "text of the conversation" (required)
  }

  Args:
    request (flask.Request): The Flask request object containing the POST data.

  Returns:
    flask.Response: A JSON response containing a list of identified subjects and their associated keywords. The response structure is:
      ```json
      [
        {
          "subject": "string", 
          "keywords": ["list of strings"]
        },
        ...
      ]
      ```
  """

  conversation = request.json.get("conversation")
  if not conversation:
    return "Error: No conversation parameter supplied in the request body", 400

  prompt = build_prompt(INSTRUCTION, conversation)

  output = generator(prompt, do_sample=True, max_new_tokens=1000, temperature=0.2)

  # Remove the input from the output before parsing
  generated_text = output[0]["generated_text"]
  generated_text = generated_text.replace(prompt, "")

  # Parse JSON
  parsed_output = extract_json(output[0]["generated_text"])

  return jsonify(parsed_output)


if __name__ == "__main__":
  app.run(debug=True, host="0.0.0.0", port=5005)