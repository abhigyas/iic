#!/usr/bin/env python3
"""
CPU-Compatible Model Quantization Script
Works on macOS without CUDA requirements
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os
import json

def quantize_model_cpu(model_path, output_path, dtype=torch.float16):
    """
    Quantize model for CPU inference using PyTorch's native quantization
    """
    print(f"Loading model from: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model in float32 first
    print("Loading model in float32...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    
    # Convert to half precision
    print("Converting to half precision...")
    model = model.half()
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save model
    print(f"Saving quantized model to: {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Add a note about the quantization
    quantization_info = {
        "quantization_method": "half_precision",
        "original_dtype": "float32",
        "quantized_dtype": "float16",
        "device": "cpu",
        "size_reduction": "~50%"
    }
    
    with open(os.path.join(output_path, "quantization_info.json"), "w") as f:
        json.dump(quantization_info, f, indent=2)
    
    print("CPU quantization complete!")
    return True

def quantize_model_dynamic(model_path, output_path):
    """
    Apply dynamic quantization (PyTorch native)
    """
    print(f"Loading model from: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    
    # Apply dynamic quantization
    print("Applying dynamic quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save model
    print(f"Saving quantized model to: {output_path}")
    quantized_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Add quantization info
    quantization_info = {
        "quantization_method": "dynamic_quantization",
        "original_dtype": "float32",
        "quantized_dtype": "qint8",
        "device": "cpu",
        "size_reduction": "~75%"
    }
    
    with open(os.path.join(output_path, "quantization_info.json"), "w") as f:
        json.dump(quantization_info, f, indent=2)
    
    print("Dynamic quantization complete!")
    return True

def check_model_size(model_path):
    """Check the size of the model"""
    total_size = 0
    
    if os.path.isdir(model_path):
        for root, dirs, files in os.walk(model_path):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
    
    size_mb = total_size / (1024 * 1024)
    size_gb = size_mb / 1024
    
    print(f"Model size: {size_mb:.2f} MB ({size_gb:.2f} GB)")
    return size_mb

def main():
    parser = argparse.ArgumentParser(description="CPU-compatible model quantization")
    parser.add_argument("--model_path", required=True, help="Path to the original model")
    parser.add_argument("--output_path", required=True, help="Path to save quantized model")
    parser.add_argument("--method", choices=["half", "dynamic"], default="half", 
                       help="Quantization method: half (float16) or dynamic (int8)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model path does not exist: {args.model_path}")
        return
    
    print("=== CPU-Compatible Model Quantization ===")
    print(f"Input model: {args.model_path}")
    print(f"Output path: {args.output_path}")
    print(f"Method: {args.method}")
    
    # Check original size
    print("\n--- Original Model Size ---")
    original_size = check_model_size(args.model_path)
    
    # Quantize
    print(f"\n--- Quantizing with {args.method} method ---")
    try:
        if args.method == "half":
            success = quantize_model_cpu(args.model_path, args.output_path)
        else:
            success = quantize_model_dynamic(args.model_path, args.output_path)
        
        if success:
            print("\n--- Quantized Model Size ---")
            quantized_size = check_model_size(args.output_path)
            
            reduction = ((original_size - quantized_size) / original_size) * 100
            print(f"Size reduction: {reduction:.1f}%")
            
            if quantized_size <= 100:
                print("✓ Model is small enough for GitHub!")
            elif quantized_size <= 1000:
                print("⚠️  Model is larger than 100MB but could work with Git LFS")
            else:
                print("⚠️  Model is still large, consider external hosting")
                
    except Exception as e:
        print(f"Error during quantization: {e}")

if __name__ == "__main__":
    main()
