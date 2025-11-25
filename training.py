import os

# Add this with your other environment variables at the top
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Check for MPS (Apple Silicon GPU) availability
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Configuration for different memory sizes
# Set to "8gb" or "24gb" based on your system
MEMORY_CONFIG = "24gb"  # Change to "8gb" if needed

if MEMORY_CONFIG == "24gb":
    print("🚀 High-quality training mode for 24GB systems")
    ENABLE_GRADIENT_CHECKPOINTING = False  # Disable for faster training
    MAX_LENGTH = 256  # Longer sequences = better context understanding
    TRAIN_BATCH_SIZE = 4  # Larger batches = more stable gradients
    EVAL_BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 4 * 2 = 8
    NUM_EPOCHS = 3  # More epochs for better learning
    MAX_STEPS = None  # Train full epochs
else:
    print("💾 Memory-optimized mode for 8GB systems")
    ENABLE_GRADIENT_CHECKPOINTING = True
    MAX_LENGTH = 32
    TRAIN_BATCH_SIZE = 1
    EVAL_BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 4
    NUM_EPOCHS = 1
    MAX_STEPS = 5

# Load model
model_id = "meta-llama/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    use_cache=True if not ENABLE_GRADIENT_CHECKPOINTING else False,
    device_map="auto" if device == "mps" else None,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Move model to MPS device if available
if device == "mps" and not hasattr(model, "hf_device_map"):
    model = model.to(device)
    print("Model moved to MPS (Apple Silicon GPU)")
else:
    print(f"Model on device: {device}")

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
        max_length=MAX_LENGTH,  # Configurable based on memory
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
if ENABLE_GRADIENT_CHECKPOINTING:
    model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled (trades compute for memory)")
else:
    print("Gradient checkpointing disabled (faster training with 24GB)")

print(f"\n📊 Training Configuration:")
print(f"   - Batch size: {TRAIN_BATCH_SIZE}")
print(f"   - Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
print(f"   - Effective batch size: {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"   - Max sequence length: {MAX_LENGTH}")
print(f"   - Epochs: {NUM_EPOCHS}")
print(f"   - Gradient checkpointing: {ENABLE_GRADIENT_CHECKPOINTING}\n")

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=20,  # More frequent evaluation for better monitoring
    logging_steps=10,  # More frequent logging
    save_steps=50,  # Save checkpoints more frequently
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    fp16=False,  # Keep False for MPS
    report_to="none",
    log_level="info",
    learning_rate=2e-5,  # Slightly higher LR for better convergence with larger batches
    max_grad_norm=1.0,  # Gradient clipping
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    gradient_checkpointing=ENABLE_GRADIENT_CHECKPOINTING,
    max_steps=MAX_STEPS,  # None for 24GB (train full epochs)
    optim="adamw_torch",
    dataloader_pin_memory=False,  # Set to False for MPS
    dataloader_num_workers=0,  # Keep at 0 for MPS
    remove_unused_columns=True,
    warmup_steps=10,  # Learning rate warmup for better training
    save_total_limit=3,  # Keep only last 3 checkpoints
    load_best_model_at_end=True,  # Load best model based on eval loss
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
)

# Memory management before training
if hasattr(torch.mps, "empty_cache"):
    torch.mps.empty_cache()
    print("MPS cache cleared")

# Training info
if device == "mps":
    if MEMORY_CONFIG == "24gb":
        print("\n✨ High-Quality Training Mode (24GB)")
        print("   - Larger batches for stable gradients")
        print("   - Longer sequences for better context")
        print("   - Multiple epochs for thorough learning")
        print("   - Best model checkpointing enabled")
    else:
        print("\n⚠️  Memory-Optimized Mode (8GB)")
        print("   - Close other applications to free up memory")
        print("   - Monitor Activity Monitor for memory pressure")
        print("   - If you get OOM errors, reduce max_length or batch size\n")

# Train the model
print("🚀 Starting training...")
if MEMORY_CONFIG == "24gb":
    print(
        "   Training with high-quality settings - this will take longer but produce better results"
    )
else:
    print("   Training with memory-optimized settings - be patient!")

try:
    trainer.train()
    print("\n✅ Training completed!")
    if MEMORY_CONFIG == "24gb":
        print("   Best model saved based on evaluation loss")
except RuntimeError as e:
    if "out of memory" in str(e).lower() or "mps" in str(e).lower():
        print("\n❌ Out of memory error!")
        print("Try these solutions:")
        if MEMORY_CONFIG == "24gb":
            print("1. Reduce TRAIN_BATCH_SIZE (currently 4)")
            print("2. Reduce MAX_LENGTH (currently 256)")
            print(
                "3. Enable gradient checkpointing (set ENABLE_GRADIENT_CHECKPOINTING = True)"
            )
            print("4. Reduce GRADIENT_ACCUMULATION_STEPS (currently 2)")
        else:
            print("1. Close other applications")
            print("2. Reduce max_length in tokenize_function")
            print("3. Reduce gradient_accumulation_steps")
            print("4. Consider using LoRA fine-tuning instead")
        raise
    else:
        raise

# Save the model and tokenizer
trainer.save_model("./fine-tuned-model")
tokenizer.save_pretrained("./fine-tuned-model")
