# Model Quantization Guide

This guide explains how to quantize your cybersecurity threat mitigation model to make it small enough for GitHub deployment.

## Current Model Size
Your model is approximately 6GB, which exceeds GitHub's limits:
- Individual file limit: 100MB
- Repository size limit: 1GB

## Quantization Options

### Option 1: BitsAndBytes Quantization (Recommended for beginners)

1. **Install dependencies:**
```bash
pip install -r quantization_requirements.txt
```

2. **4-bit quantization (smallest size):**
```bash
python quantize_model.py --model_path /Users/abhigyasinha/Desktop/cybersecurity-threat-mitigation-instruct --output_path ./quantized_model_4bit --bits 4
```

3. **8-bit quantization (better quality, larger size):**
```bash
python quantize_model.py --model_path /Users/abhigyasinha/Desktop/cybersecurity-threat-mitigation-instruct --output_path ./quantized_model_8bit --bits 8
```

**Expected size reduction:** 75-85% smaller

### Option 2: GGUF Format (Best compression)

1. **Install llama-cpp-python:**
```bash
pip install llama-cpp-python
```

2. **Convert to GGUF:**
```bash
python convert_to_gguf.py
```

**Expected size reduction:** 80-90% smaller

### Option 3: AutoGPTQ (Good balance)

1. **Install dependencies:**
```bash
pip install auto-gptq
```

2. **Quantize with GPTQ:**
```bash
python quantize_gptq.py --model_path /Users/abhigyasinha/Desktop/cybersecurity-threat-mitigation-instruct --output_path ./quantized_model_gptq --bits 4
```

**Expected size reduction:** 70-80% smaller

## Loading Quantized Models

### For BitsAndBytes quantized models:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Load quantized model
tokenizer = AutoTokenizer.from_pretrained("./quantized_model_4bit")
model = AutoModelForCausalLM.from_pretrained(
    "./quantized_model_4bit",
    device_map="auto",
    torch_dtype=torch.float16
)
```

### For GGUF models:
```python
from llama_cpp import Llama

# Load GGUF model
llm = Llama(model_path="./cybersecurity_model_quantized.gguf")

# Generate text
output = llm("What is a cybersecurity threat?", max_tokens=256)
```

### For GPTQ models:
```python
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer

# Load GPTQ model
tokenizer = AutoTokenizer.from_pretrained("./quantized_model_gptq")
model = AutoGPTQForCausalLM.from_quantized("./quantized_model_gptq")
```

## Recommendations

1. **Start with Option 1 (BitsAndBytes)** - it's the most straightforward and well-supported.

2. **If you need maximum compression**, try Option 2 (GGUF format).

3. **For production use**, consider hosting larger models on:
   - Hugging Face Hub (free for public models)
   - AWS S3 or similar cloud storage
   - Git LFS for larger files

## Alternative: Model Hosting

If quantization doesn't reduce the size enough:

1. **Hugging Face Hub:**
   - Upload your model to Hugging Face
   - Load it dynamically in your application
   - Free for public models

2. **Git LFS:**
   - Use Git Large File Storage for models > 100MB
   - GitHub supports Git LFS up to 1GB

3. **External hosting:**
   - Host model files separately
   - Download them when your application starts

## Testing Your Quantized Model

After quantization, test your model to ensure quality:

```python
# Test script
def test_quantized_model(model_path):
    # Load your quantized model
    # Run inference on sample cybersecurity scenarios
    # Compare outputs with original model
    pass
```

Choose the quantization method that best balances size reduction with model quality for your specific use case.
