#!/usr/bin/env python3
"""
Convert model to GGUF format for maximum compression
GGUF is a more efficient format for inference
"""

import os
import subprocess
import sys

def convert_to_gguf(model_path, output_path, quantization_level="q4_0"):
    """
    Convert model to GGUF format using llama.cpp
    
    Quantization levels:
    - q4_0: 4-bit quantization (smallest, good quality)
    - q4_1: 4-bit quantization (alternative)
    - q5_0: 5-bit quantization (balanced)
    - q5_1: 5-bit quantization (alternative)
    - q8_0: 8-bit quantization (larger but better quality)
    - f16: 16-bit float (largest but best quality)
    """
    
    print("Converting model to GGUF format...")
    print(f"Input: {model_path}")
    print(f"Output: {output_path}")
    print(f"Quantization: {quantization_level}")
    
    # First, we need to convert to GGUF format
    temp_gguf = os.path.join(os.path.dirname(output_path), "temp_model.gguf")
    
    # Convert to GGUF (you'll need llama.cpp installed)
    convert_cmd = [
        "python", "-m", "llama_cpp.convert",
        "--model", model_path,
        "--outfile", temp_gguf,
        "--vocab-type", "spm"
    ]
    
    try:
        subprocess.run(convert_cmd, check=True)
        print("✓ Converted to GGUF format")
    except subprocess.CalledProcessError as e:
        print(f"Error converting to GGUF: {e}")
        return False
    
    # Quantize the GGUF file
    quantize_cmd = [
        "python", "-m", "llama_cpp.quantize",
        temp_gguf,
        output_path,
        quantization_level
    ]
    
    try:
        subprocess.run(quantize_cmd, check=True)
        print(f"✓ Quantized to {quantization_level}")
        
        # Remove temporary file
        os.remove(temp_gguf)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error quantizing: {e}")
        return False

def main():
    model_path = "/Users/abhigyasinha/Desktop/cybersecurity-threat-mitigation-instruct"
    output_path = "/Users/abhigyasinha/Documents/GitHub/iic/cybersecurity_model_quantized.gguf"
    
    # You can change the quantization level here
    # q4_0 gives the best size/quality tradeoff
    convert_to_gguf(model_path, output_path, "q4_0")
    
    # Check file size
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✓ Quantized model size: {size_mb:.2f} MB")
        
        if size_mb > 100:
            print("⚠️  Model is still larger than 100MB (GitHub limit)")
            print("Consider using q4_0 or hosting elsewhere")
        else:
            print("✓ Model is small enough for GitHub!")

if __name__ == "__main__":
    main()
