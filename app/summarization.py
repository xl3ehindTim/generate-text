import os
import json

import logging
import torch

from transformers import pipeline


def extract_json(s):
    s = s[next(idx for idx, c in enumerate(s) if c in "{["):]
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        return json.loads(s[:e.pos])


model_name = "mistralai/Mistral-7B-Instruct-v0.2"
subject_extraction = pipeline(
    "summarization", model=model_name, device_map="auto"
)

conversation = "Can you summarize this conversation in one sentence, highlighting the key point or topic discussed? Here is the conversation: Whats ur band called? its called 1 idiot, 1 bass guitar"

output = subject_extraction(conversation, max_length=250, truncation=True)
parsed = extract_json(output[0]["summary_text"])

print(parsed)