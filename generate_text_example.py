import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Print available files in the model directory
model_id = "fine-tuned-model"
print("Files in model directory:", os.listdir(model_id))

# Load model with additional safety checks
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        use_cache=True,
        local_files_only=True,  # Ensure we're using local files
    )

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # Test the model with a very simple input first
    test_input = "Why is pluto not a planet?"
    inputs = tokenizer(
        test_input,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
        return_attention_mask=True,  # Explicitly request attention mask
    )

    print("Model loaded successfully")
    print("Vocab size:", len(tokenizer))
    print("Model parameters:", sum(p.numel() for p in model.parameters()))

except Exception as e:
    print(f"Error loading model: {str(e)}")

# Prepare the prompt
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Why is pluto not a planet?"},
]

# Convert messages to a single string
prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

# Tokenize input
inputs = tokenizer(
    prompt, return_tensors="pt", padding=True, truncation=True, max_length=128
)

# Generate with more conservative parameters
try:
    # Move everything to CPU explicitly
    model = model.cpu()
    inputs = {k: v.cpu() for k, v in inputs.items()}

    # Generate with more stable settings
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=32,  # Reduced for testing
            do_sample=False,  # Use greedy decoding for testing
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,  # Add repetition penalty
        )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nGenerated response:", generated_text)

except Exception as e:
    print(f"Error during generation: {str(e)}")
    print(
        "Model output shape:",
        outputs.shape if "outputs" in locals() else "No outputs generated",
    )
    print("Input shape:", inputs.input_ids.shape)
