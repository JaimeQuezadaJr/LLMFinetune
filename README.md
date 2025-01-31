# Llama Text Generation Example

This is a simple Python script demonstrating how to use Hugging Face's transformers library to generate text using Meta's Llama model.

## Overview

The script (`generate_text_example.py`) shows how to:
- Load a Llama model from Hugging Face's model hub
- Set up a text generation pipeline
- Generate responses using a chat-style format

## Requirements

The project requires several Python packages, primarily:
- transformers
- torch
- accelerate

All dependencies are listed in `requirements.txt`. Install them using:

```
pip install -r requirements.txt

## Usage

The script uses Meta's Llama 3.2 1B Instruct model to generate text responses. It demonstrates:
1. Setting up a text generation pipeline
2. Formatting messages in a chat-style format
3. Generating responses with specified parameters

Example code:

```python
from transformers import pipeline

model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline("text-generation", model=model_id, device_map="auto")
```

The `device_map="auto"` parameter automatically handles model placement on available hardware (CPU/GPU).

## Parameters

The script uses the following generation parameters:
- `max_new_tokens=128`: Limits the length of generated text
- `do_sample=True`: Enables sampling for more diverse outputs

## Note

Make sure you have appropriate access rights to the Llama model on Hugging Face's model hub, as some models require acceptance of terms of use or special access permissions.