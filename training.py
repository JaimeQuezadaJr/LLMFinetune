import os

# Add this with your other environment variables at the top
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load the base model and tokenizer
model_id = "meta-llama/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float32, use_cache=False
)  # Must be float32 for MacBooks!
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Load the training dataset
dataset = load_dataset("csv", data_files="planet_qa_dataset.csv", split="train")


# Define a function to apply the chat template
def apply_chat_template(example):
    messages = [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"prompt": prompt}


# Apply the chat template function to the dataset
new_dataset = dataset.map(apply_chat_template)
new_dataset = new_dataset.train_test_split(
    0.05
)  # Let's keep 5% of the data for testing


# Tokenize the data
def tokenize_function(example):
    tokens = tokenizer(
        example["prompt"],
        padding="max_length",
        truncation=True,
        max_length=32,  # Keep this small
    )
    tokens["labels"] = [
        -100 if token == tokenizer.pad_token_id else token
        for token in tokens["input_ids"]
    ]
    return tokens


# Apply tokenize_function to each row
tokenized_dataset = new_dataset.map(tokenize_function)
tokenized_dataset = tokenized_dataset.remove_columns(["question", "answer", "prompt"])

# Define training arguments
model.train()
model.gradient_checkpointing_enable()
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=5,
    logging_steps=5,
    save_steps=150,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,  # Reduced from 4 to 1
    fp16=False,
    report_to="none",
    log_level="info",  # Change to "debug" to see more information
    learning_rate=1e-5,
    max_grad_norm=2,
    gradient_accumulation_steps=4,  # Reduced from 8 to 4
    gradient_checkpointing=True,
    max_steps=5,  # Reduced from 10 to 5 for initial test
    optim="adamw_torch",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
)

# Add this before training starts
if hasattr(torch.mps, "empty_cache"):
    torch.mps.empty_cache()

# Train the model
print("Starting training...")  # Add this to confirm we reach this point
trainer.train()
print("Training completed!")  # Add this to confirm if training finishes

# Save the model and tokenizer
trainer.save_model("./fine-tuned-model")
tokenizer.save_pretrained("./fine-tuned-model")
