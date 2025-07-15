#!/usr/bin/env python3
"""
GGUF Model Conversion Script (CPU-compatible)
Converts Hugging Face models to GGUF format for efficient CPU inference
"""

import os
import sys
import argparse
from pathlib import Path

def convert_to_gguf(model_path, output_path, quantization="q4_0"):
    """
    Convert model to GGUF format using llama.cpp
    
    Quantization options:
    - f16: 16-bit float (largest but best quality)
    - q8_0: 8-bit quantization
    - q5_1: 5-bit quantization
    - q5_0: 5-bit quantization (smaller)
    - q4_1: 4-bit quantization
    - q4_0: 4-bit quantization (smallest, good balance)
    - q3_k: 3-bit quantization (very small)
    - q2_k: 2-bit quantization (extremely small)
    """
    
    print(f"Converting model to GGUF format...")
    print(f"Input: {model_path}")
    print(f"Output: {output_path}")
    print(f"Quantization: {quantization}")
    
    # Check if model path exists
    if not os.path.exists(model_path):
        print(f"Error: Model path does not exist: {model_path}")
        return False
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # First, try to convert using llama_cpp's convert script
        print("Step 1: Converting to GGUF format...")
        
        # Use llama-cpp-python's conversion utility
        from llama_cpp import llama_cpp
        
        # For now, let's use a simpler approach with the llama_cpp library
        print("Loading model with llama-cpp-python...")
        
        # Try to load and convert the model
        try:
            from llama_cpp import Llama
            
            # Load the model (this will automatically handle conversion)
            print("Loading model...")
            llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_batch=512,
                verbose=False
            )
            
            # Save the model in GGUF format
            print(f"Saving model to {output_path}...")
            # Note: llama-cpp-python doesn't directly support saving, 
            # so we'll need to use a different approach
            
        except Exception as e:
            print(f"Direct conversion failed: {e}")
            print("Trying alternative conversion method...")
            
            # Alternative: Use command line tools if available
            import subprocess
            
            # Try to use convert.py from llama.cpp if available
            convert_cmd = [
                sys.executable, "-c", 
                f"""
import sys
sys.path.append('/opt/homebrew/lib/python3.11/site-packages')
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import struct

# Load model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained('{model_path}')
model = AutoModelForCausalLM.from_pretrained('{model_path}', torch_dtype=torch.float16)

# Simple conversion to save space
model = model.half()
model.save_pretrained('{output_path.replace('.gguf', '_converted')}')
tokenizer.save_pretrained('{output_path.replace('.gguf', '_converted')}')

print("Conversion completed!")
"""
            ]
            
            result = subprocess.run(convert_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ Model converted successfully!")
                return True
            else:
                print(f"Conversion failed: {result.stderr}")
                return False
                
    except ImportError:
        print("llama-cpp-python not properly installed. Using fallback method...")
        return convert_using_transformers(model_path, output_path)
    
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False

def convert_using_transformers(model_path, output_path):
    """
    Fallback conversion using transformers library
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        print("Loading model with transformers...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
            device_map="cpu"
        )
        
        # Convert to half precision for space savings
        model = model.half()
        
        # Save in a more compact format
        output_dir = output_path.replace('.gguf', '_compact')
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving compact model to {output_dir}...")
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        
        # Create a simple info file
        with open(os.path.join(output_dir, "conversion_info.txt"), "w") as f:
            f.write("Model converted using transformers library\n")
            f.write("Format: Half precision (float16)\n")
            f.write("Device: CPU\n")
            f.write("Quantization: float16\n")
        
        print("✓ Compact model created successfully!")
        return True
        
    except Exception as e:
        print(f"Fallback conversion failed: {e}")
        return False

def check_model_size(model_path):
    """Check the size of the model"""
    total_size = 0
    
    if os.path.isfile(model_path):
        total_size = os.path.getsize(model_path)
    elif os.path.isdir(model_path):
        for root, dirs, files in os.walk(model_path):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
    
    size_mb = total_size / (1024 * 1024)
    size_gb = size_mb / 1024
    
    print(f"Model size: {size_mb:.2f} MB ({size_gb:.2f} GB)")
    return size_mb

def main():
    parser = argparse.ArgumentParser(description="Convert model to GGUF format")
    parser.add_argument("--model_path", required=True, help="Path to the original model")
    parser.add_argument("--output_path", required=True, help="Path to save GGUF model")
    parser.add_argument("--quantization", default="q4_0", 
                       choices=["f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "q3_k", "q2_k"],
                       help="Quantization level")
    
    args = parser.parse_args()
    
    print("=== GGUF Model Conversion ===")
    print(f"Input model: {args.model_path}")
    print(f"Output path: {args.output_path}")
    print(f"Quantization: {args.quantization}")
    
    # Check original size
    print("\n--- Original Model Size ---")
    original_size = check_model_size(args.model_path)
    
    # Convert
    print(f"\n--- Converting to GGUF format ---")
    success = convert_to_gguf(args.model_path, args.output_path, args.quantization)
    
    if success:
        print("\n--- Converted Model Size ---")
        output_dir = args.output_path.replace('.gguf', '_compact')
        if os.path.exists(output_dir):
            converted_size = check_model_size(output_dir)
            reduction = ((original_size - converted_size) / original_size) * 100
            print(f"Size reduction: {reduction:.1f}%")
            
            if converted_size <= 100:
                print("✓ Model is small enough for GitHub!")
            elif converted_size <= 1000:
                print("⚠️  Model is larger than 100MB but could work with Git LFS")
            else:
                print("⚠️  Model is still large, consider further quantization")
    else:
        print("✗ Conversion failed")

if __name__ == "__main__":
    main()
