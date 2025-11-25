# LLM Fine-Tuning Project

This project demonstrates how to fine-tune Meta's Llama 3.2-1B-Instruct model on a custom dataset using Hugging Face's transformers library, and then use the fine-tuned model for text generation.

## How to Run This Project

### Prerequisites

1. **Python 3.8+** installed on your system
2. **Hugging Face Account** with access to Llama models:
   - Create an account at [huggingface.co](https://huggingface.co)
   - Accept the terms of use for `meta-llama/Llama-3.2-1B-Instruct` at [this page](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
   - Generate an access token at [Settings > Access Tokens](https://huggingface.co/settings/tokens)
   - Login using: `huggingface-cli login` (or set `HF_TOKEN` environment variable)

### Step-by-Step Instructions

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including `transformers`, `torch`, `datasets`, and `accelerate`.

#### 2. Set Up Hugging Face Authentication

If you haven't already, authenticate with Hugging Face:

```bash
huggingface-cli login
```

Enter your access token when prompted. Alternatively, you can set an environment variable:

```bash
export HF_TOKEN=your_token_here
```

#### 3. Train the Model

Run the training script to fine-tune the Llama model on the planet Q&A dataset:

```bash
python training.py
```

**What happens:**
- Downloads the base `meta-llama/Llama-3.2-1B-Instruct` model (first time only)
- Loads `planet_qa_dataset.csv` (100 Q&A pairs)
- Fine-tunes the model for 1 epoch
- Saves the fine-tuned model to `./fine-tuned-model/`

**Expected output:**
- Training progress logs
- Evaluation metrics
- Model saved to `fine-tuned-model/` directory

**Note:** Training may take several minutes depending on your hardware. The script is configured for MacBooks (CPU/MPS).

#### 4. Test the Fine-Tuned Model

Once training is complete, test the model:

```bash
python generate_text_example.py
```

**What happens:**
- Loads the fine-tuned model from `./fine-tuned-model/`
- Displays model information (vocab size, parameters)
- Generates a response to "Why is pluto not a planet?"

**Expected output:**
```
Files in model directory: [...]
Model loaded successfully
Vocab size: ...
Model parameters: ...

Generated response: ...
```

### Troubleshooting

**Issue: "Model not found" or authentication errors**
- Make sure you've accepted the terms for Llama 3.2-1B-Instruct on Hugging Face
- Verify your access token is valid: `huggingface-cli whoami`
- Try logging in again: `huggingface-cli login`

**Issue: "fine-tuned-model directory not found"**
- Make sure you've run `training.py` first
- Check that training completed successfully
- Verify the `fine-tuned-model/` directory exists in the project root

**Issue: Out of memory errors during training**
- Reduce `per_device_train_batch_size` in `training.py`
- Reduce `max_length` in the tokenization function
- Close other applications to free up memory

**Issue: Training is very slow**
- This is normal for CPU training
- The script uses conservative settings (1 epoch, small batch size) to work on various hardware
- Consider using a GPU if available (modify `device_map` settings)

## Project Overview

This project follows a complete machine learning workflow:

1. **Training Phase** (`training.py`): Fine-tunes Llama 3.2-1B-Instruct on a planet Q&A dataset
2. **Inference Phase** (`generate_text_example.py`): Tests the fine-tuned model with sample questions

### Workflow

```
planet_qa_dataset.csv → training.py → fine-tuned-model/ → generate_text_example.py → Generated Text
```

## Project Structure

- **`training.py`**: Script that fine-tunes the Llama model on the planet Q&A dataset
- **`original_training.py`**: Original training script (references different dataset)
- **`generate_text_example.py`**: Script that loads and tests the fine-tuned model
- **`planet_qa_dataset.csv`**: Training dataset with 100 question-answer pairs about planets
- **`fine-tuned-model/`**: Directory where the fine-tuned model is saved (created after training)

## Training Process (`training.py`)

The training script performs the following steps:

1. **Loads the base model**: Downloads `meta-llama/Llama-3.2-1B-Instruct` from Hugging Face Hub
2. **Loads the dataset**: Reads `planet_qa_dataset.csv` with question-answer pairs
3. **Applies chat template**: Formats data using Llama's chat template format
4. **Tokenizes data**: Converts text to token IDs with proper padding and truncation
5. **Trains the model**: Fine-tunes using Hugging Face's `Trainer` class
6. **Saves the model**: Exports fine-tuned model to `./fine-tuned-model`

### Training Configuration

- **Base Model**: `meta-llama/Llama-3.2-1B-Instruct` (1 billion parameters)
- **Dataset**: 100 Q&A pairs about planets (95% train, 5% test split)
- **Training Epochs**: 1 epoch
- **Learning Rate**: 1e-5
- **Batch Size**: 1 (with gradient accumulation of 4)
- **Max Length**: 32 tokens
- **Optimizer**: AdamW

### Running Training

```bash
python training.py
```

**Note**: You'll need access to the Llama model on Hugging Face Hub. Some models require accepting terms of use.

## Inference Process (`generate_text_example.py`)

This script loads and tests the fine-tuned model. It performs the following operations:

1. **Loads a locally fine-tuned model** from the `fine-tuned-model` directory
2. **Initializes the tokenizer and model** using Hugging Face's `AutoTokenizer` and `AutoModelForCausalLM`
3. **Configures tokenizer settings** (sets padding token if missing)
4. **Tests the model** with a simple input to verify it's working
5. **Generates text** using a chat-style prompt format

### Running Inference

```bash
python generate_text_example.py
```

The script will:
- List files in the model directory
- Load the model and tokenizer
- Display model information (vocab size, parameter count)
- Generate a response to the question "Why is pluto not a planet?"

### Generation Parameters

- **`max_new_tokens=32`**: Limits the length of generated text to 32 new tokens
- **`do_sample=False`**: Uses greedy decoding (deterministic, always picks most likely token)
- **`repetition_penalty=1.2`**: Reduces repetition in generated text
- **Device**: Explicitly runs on CPU (`model.cpu()`)

## Complete Workflow

### Step 1: Prepare Dataset
The `planet_qa_dataset.csv` file contains 100 question-answer pairs about planets. Each row has:
- `question`: A question about planets (e.g., "Why is Pluto not a planet?")
- `answer`: The corresponding answer (e.g., "Pluto was reclassified as a dwarf planet in 2006...")

### Step 2: Fine-Tune the Model
Run `training.py` to:
- Load the base Llama 3.2-1B-Instruct model
- Fine-tune it on the planet Q&A dataset
- Save the fine-tuned model to `./fine-tuned-model/`

### Step 3: Test the Fine-Tuned Model
Run `generate_text_example.py` to:
- Load the fine-tuned model from `./fine-tuned-model/`
- Test it with the question "Why is pluto not a planet?"
- Generate and display the model's response

## How It Uses Hugging Face

### Training (`training.py`)
- **`AutoModelForCausalLM`**: Loads the base Llama model
- **`AutoTokenizer`**: Loads the tokenizer
- **`load_dataset`**: Loads CSV data using Hugging Face datasets
- **`Trainer`**: Handles the training loop, optimization, and evaluation
- **`TrainingArguments`**: Configures training hyperparameters

### Inference (`generate_text_example.py`)
- **`AutoTokenizer.from_pretrained()`**: Loads tokenizer from local files
- **`AutoModelForCausalLM.from_pretrained()`**: Loads fine-tuned model from local files
- **`tokenizer()`**: Converts text to token IDs
- **`model.generate()`**: Generates text using the fine-tuned model
- **`tokenizer.decode()`**: Converts token IDs back to text

## Requirements

The project requires several Python packages, primarily:
- `transformers`: Hugging Face's transformers library
- `torch`: PyTorch for model operations
- `datasets`: Hugging Face datasets library for data loading
- `accelerate`: For efficient model loading

All dependencies are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

## Quick Start

1. **Train the model**:
   ```bash
   python training.py
   ```
   This will create the `fine-tuned-model/` directory.

2. **Test the model**:
   ```bash
   python generate_text_example.py
   ```

## Notes

- **Model Access**: Make sure you have appropriate access rights to the Llama model on Hugging Face's model hub, as some models require acceptance of terms of use or special access permissions.
- **Hardware**: The training script is configured for MacBooks (uses `torch.float32`, `fp16=False`). Adjust settings for other hardware.
- **Local Model**: The inference script (`generate_text_example.py`) requires a **locally fine-tuned model** in the `fine-tuned-model/` directory. Make sure this directory contains all necessary model files (typically `config.json`, `pytorch_model.bin` or `model.safetensors`, `tokenizer.json`, etc.).