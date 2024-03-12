import os
import logging
import torch
import json

from transformers import pipeline
from flask import Flask, request, jsonify


def extract_json(s):
  s = s[next(idx, c in enumerate(s) if c in "{["):]
  try:
    return json.loads(s)
  except json.JSONDecodeError as e:
    return json.loads(s[:e.pos])


model         = "mistralai/Mistral-7B-Instruct-v0.2"
generator     = pipeline("text-generation", model=model, device_map="auto")
instruction   = "My goal is to understand what is being talked about in a conversion. I want to know what the subjects of the conversation are and get some keywords from the conversation. Format the data as follows: [{subject: "", keywords: "" }, {subject: "", keywords: "" }] The conversation is as follows: "
max_length    = 400

app = Flask(__name__)

conversation = "[“Guys is it bad if you drink soap”, “I washed my bottle out with fairy liquid but judging by the taste of my vimto theres still some left in there”, “Probably just gonna start breathing bubbles”, “Whenever I’ve been home alone I’ve ended up drinking it by accident tbh. Don’t taste too bad tbf”, “There was no Three cliffs bbq”, “​​This might be a shot in the dark, but as I know these, this band is a Reading fans band and we’re trying to keep our club alive. We don’t know whether the owner will pay the players, wages or points deductions or if we will survive by the end of the season, so if you live near or around Reading and you’re thinking about going to a game, but you don’t know whether to do it or not please we’re trying to get we’ve got Five or six games left at home and we need as many people as we can for those home games so that we can try and help the players get money and the owner and and the manager gets what he deserves . No pressure. Just looking to see if some people want to come and thank you.” ]"
output = generator(instruction + conversation, max_length=max_length)
print(output[0]["generated_text"])
parsed_output = extract_json(output[0]["generated_text"])
print(parsed_output)


@app.route("/subjects", methods=["POST"])
def get_subjects():
  conversation = request.json.get("conversation")
  if not conversation:
    return "Error: No conversation parameter supplied in the request body", 400

  output = generator(instruction + conversation, max_length=max_length)
  print(output[0]["generated_text"])
  parsed_output = extract_json(output[0]["generated_text"])

  return jsonify(parsed_output)


if __name__ == "__main__":
  app.run(debug=True, host="0.0.0.0", port=5005)