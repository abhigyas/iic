#!/usr/bin/env python3
"""
Test script for quantized models
"""

import torch
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time

def test_bitsandbytes_model(model_path):
    """Test BitsAndBytes quantized model"""
    print(f"Testing BitsAndBytes model: {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Test inference
        test_prompt = "What is a cybersecurity threat?"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=150,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        inference_time = time.time() - start_time
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Inference time: {inference_time:.2f} seconds")
        print(f"✓ Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing model: {e}")
        return False

def test_gguf_model(model_path):
    """Test GGUF model"""
    print(f"Testing GGUF model: {model_path}")
    
    try:
        from llama_cpp import Llama
        
        llm = Llama(model_path=model_path)
        
        test_prompt = "What is a cybersecurity threat?"
        
        start_time = time.time()
        output = llm(test_prompt, max_tokens=100, temperature=0.7)
        inference_time = time.time() - start_time
        
        print(f"✓ GGUF model loaded successfully")
        print(f"✓ Inference time: {inference_time:.2f} seconds")
        print(f"✓ Response: {output['choices'][0]['text']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing GGUF model: {e}")
        return False

def check_model_size(model_path):
    """Check the size of the model"""
    total_size = 0
    
    if os.path.isfile(model_path):
        # Single file (GGUF)
        total_size = os.path.getsize(model_path)
    else:
        # Directory with multiple files
        for root, dirs, files in os.walk(model_path):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
    
    size_mb = total_size / (1024 * 1024)
    size_gb = size_mb / 1024
    
    print(f"Model size: {size_mb:.2f} MB ({size_gb:.2f} GB)")
    
    if size_mb <= 100:
        print("✓ Model is small enough for GitHub!")
    elif size_mb <= 1000:
        print("⚠️  Model is larger than 100MB but could work with Git LFS")
    else:
        print("✗ Model is too large for GitHub, consider external hosting")
    
    return size_mb

def main():
    print("=== Model Quantization Test ===\n")
    
    # Test different model formats
    test_models = [
        ("./quantized_model_4bit", "bitsandbytes"),
        ("./quantized_model_8bit", "bitsandbytes"),
        ("./cybersecurity_model_quantized.gguf", "gguf"),
        ("./quantized_model_gptq", "gptq")
    ]
    
    for model_path, model_type in test_models:
        if os.path.exists(model_path):
            print(f"\n--- Testing {model_type} model ---")
            size_mb = check_model_size(model_path)
            
            if model_type == "bitsandbytes":
                test_bitsandbytes_model(model_path)
            elif model_type == "gguf":
                test_gguf_model(model_path)
            elif model_type == "gptq":
                # Similar to bitsandbytes but with different loading
                test_bitsandbytes_model(model_path)
        else:
            print(f"Model not found: {model_path}")

if __name__ == "__main__":
    main()
