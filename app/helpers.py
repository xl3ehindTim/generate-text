import json


def build_prompt(instruction, conversation):
  """
  Build input format
  """
  return f"[INST] {instruction} {conversation} [/INST]"


def extract_json(s):
  """
  Extract JSON from output
  """
  s = s[next(idx for idx, c in enumerate(s) if c in "{["):]
  try:
    return json.loads(s)
  except json.JSONDecodeError as e:
    return json.loads(s[:e.pos])