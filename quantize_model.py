#!/usr/bin/env python3
"""
Model Quantization Script
Quantizes a Llama model to reduce size for GitHub deployment
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import argparse
import os
import shutil

def quantize_model_4bit(model_path, output_path):
    """
    Quantize model to 4-bit precision using BitsAndBytes
    """
    print(f"Loading model from: {model_path}")
    
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save quantized model
    print(f"Saving quantized model to: {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print("Quantization complete!")

def quantize_model_8bit(model_path, output_path):
    """
    Quantize model to 8-bit precision using BitsAndBytes
    """
    print(f"Loading model from: {model_path}")
    
    # Configure 8-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save quantized model
    print(f"Saving quantized model to: {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print("Quantization complete!")

def main():
    parser = argparse.ArgumentParser(description="Quantize a Llama model")
    parser.add_argument("--model_path", required=True, help="Path to the original model")
    parser.add_argument("--output_path", required=True, help="Path to save quantized model")
    parser.add_argument("--bits", choices=["4", "8"], default="4", help="Quantization bits (4 or 8)")
    
    args = parser.parse_args()
    
    if args.bits == "4":
        quantize_model_4bit(args.model_path, args.output_path)
    else:
        quantize_model_8bit(args.model_path, args.output_path)

if __name__ == "__main__":
    main()
