#!/usr/bin/env python3
"""
Quantize model using AutoGPTQ
This provides good compression with minimal quality loss
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import argparse
import os

def quantize_with_gptq(model_path, output_path, bits=4):
    """
    Quantize model using GPTQ
    """
    print(f"Loading model from: {model_path}")
    
    # Configure quantization
    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=128,
        desc_act=False,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model for quantization
    model = AutoGPTQForCausalLM.from_pretrained(
        model_path, 
        quantize_config=quantize_config
    )
    
    # You would typically need calibration data for GPTQ
    # For now, we'll use a simple calibration
    calibration_data = [
        "What is cybersecurity?",
        "Explain network security threats.",
        "How to prevent malware attacks?",
        "What are common vulnerabilities?",
        "Describe incident response procedures."
    ]
    
    # Quantize the model
    print("Quantizing model...")
    model.quantize(calibration_data)
    
    # Save quantized model
    print(f"Saving quantized model to: {output_path}")
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)
    
    print("GPTQ quantization complete!")

def main():
    parser = argparse.ArgumentParser(description="Quantize model using GPTQ")
    parser.add_argument("--model_path", required=True, help="Path to the original model")
    parser.add_argument("--output_path", required=True, help="Path to save quantized model")
    parser.add_argument("--bits", type=int, choices=[2, 3, 4, 8], default=4, help="Quantization bits")
    
    args = parser.parse_args()
    
    quantize_with_gptq(args.model_path, args.output_path, args.bits)

if __name__ == "__main__":
    main()
