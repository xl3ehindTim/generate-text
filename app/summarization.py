import os
import logging
import torch

from transformers import pipeline

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
subject_extraction = pipeline(
    "text-summarization", model=model_name, device=0 if torch.cuda.is_available() else -1
)

conversation = "Can you summarize this conversation in one sentence, highlighting the key point or topic discussed? Here is the conversation: Whats ur band called? its called 1 idiot, 1 bass guitar"

with torch.cuda.amp.autocast_enabled() if torch.cuda.is_available() else torch.no_grad():
    output = subject_extraction(conversation, max_length=100, truncation=True)

print("output", output)

print("-----------------------------------")

print(output[0]["summary_text"])