from transformers import pipeline

model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline("text-generation", model=model_id, device_map="auto")

messages = [
    {'role' : 'system', 'content' : 'You are a helpful assistant.'},
    {'role' : 'user', 'content' : 'What is the capital of England?'}
]

outputs = pipe(messages, max_new_tokens=128, do_sample=True)
print(outputs[0]['generated_text'][-1])
