#!/usr/bin/env python3
"""
Advanced Model Quantization Script
Performs aggressive quantization to reduce model size for GitHub deployment
"""

import os
import torch
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

def quantize_weights_int8(tensor):
    """
    Quantize weights to int8 with scale and zero point
    """
    # Calculate scale and zero point
    min_val = tensor.min()
    max_val = tensor.max()
    
    # Calculate scale
    scale = (max_val - min_val) / 255
    
    # Calculate zero point
    zero_point = torch.round(-min_val / scale).clamp(0, 255)
    
    # Quantize
    quantized = torch.round(tensor / scale + zero_point).clamp(0, 255).to(torch.uint8)
    
    return quantized, scale, zero_point

def quantize_weights_int4(tensor):
    """
    Quantize weights to int4 (4-bit)
    """
    # Calculate scale
    min_val = tensor.min()
    max_val = tensor.max()
    scale = (max_val - min_val) / 15  # 4-bit has 16 levels (0-15)
    
    # Quantize to 4-bit
    quantized = torch.round((tensor - min_val) / scale).clamp(0, 15).to(torch.uint8)
    
    return quantized, scale, min_val

def aggressive_quantization(model_path, output_path, quantization_bits=4):
    """
    Perform aggressive quantization to significantly reduce model size
    """
    print(f"Loading model from: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    
    print(f"Performing {quantization_bits}-bit quantization...")
    
    # Create quantized state dict
    quantized_state = {}
    quantization_params = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_data = param.data.clone()
            
            if quantization_bits == 4:
                quantized_weights, scale, min_val = quantize_weights_int4(param_data)
                quantization_params[name] = {
                    'scale': scale.item(),
                    'min_val': min_val.item(),
                    'shape': list(param_data.shape),
                    'bits': 4
                }
            elif quantization_bits == 8:
                quantized_weights, scale, zero_point = quantize_weights_int8(param_data)
                quantization_params[name] = {
                    'scale': scale.item(),
                    'zero_point': zero_point.item(),
                    'shape': list(param_data.shape),
                    'bits': 8
                }
            
            quantized_state[name] = quantized_weights
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save quantized model
    print(f"Saving quantized model to: {output_path}")
    
    # Save quantized weights
    torch.save(quantized_state, os.path.join(output_path, "quantized_weights.pt"))
    
    # Save quantization parameters
    with open(os.path.join(output_path, "quantization_params.json"), "w") as f:
        json.dump(quantization_params, f, indent=2)
    
    # Save model config
    model.config.save_pretrained(output_path)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_path)
    
    # Create loading script
    loader_script = f'''
import torch
import json
from transformers import AutoTokenizer, AutoConfig

def load_quantized_model(model_path):
    """Load the quantized model"""
    
    # Load quantization parameters
    with open(f"{{model_path}}/quantization_params.json", "r") as f:
        quant_params = json.load(f)
    
    # Load quantized weights
    quantized_weights = torch.load(f"{{model_path}}/quantized_weights.pt", map_location="cpu")
    
    # Load config and tokenizer
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("Quantized model loaded successfully!")
    print(f"Quantization: {{quant_params[list(quant_params.keys())[0]]['bits']}}-bit")
    
    return quantized_weights, quant_params, config, tokenizer

def dequantize_weights(quantized_weights, quant_params):
    """Dequantize weights back to float"""
    dequantized = {{}}
    
    for name, qweight in quantized_weights.items():
        params = quant_params[name]
        
        if params['bits'] == 4:
            # Dequantize 4-bit
            dequantized[name] = qweight.float() * params['scale'] + params['min_val']
        elif params['bits'] == 8:
            # Dequantize 8-bit
            dequantized[name] = (qweight.float() - params['zero_point']) * params['scale']
        
        # Reshape to original shape
        dequantized[name] = dequantized[name].reshape(params['shape'])
    
    return dequantized

# Example usage:
# quantized_weights, quant_params, config, tokenizer = load_quantized_model("./quantized_model")
# dequantized_weights = dequantize_weights(quantized_weights, quant_params)
'''
    
    with open(os.path.join(output_path, "load_quantized_model.py"), "w") as f:
        f.write(loader_script)
    
    # Create info file
    info = {
        "original_model": model_path,
        "quantization_bits": quantization_bits,
        "compression_method": "weight_quantization",
        "estimated_size_reduction": "75-85%" if quantization_bits == 4 else "50-60%",
        "usage": "Use load_quantized_model.py to load the quantized model"
    }
    
    with open(os.path.join(output_path, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    
    print("Aggressive quantization complete!")
    return True

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
    parser = argparse.ArgumentParser(description="Aggressive model quantization")
    parser.add_argument("--model_path", required=True, help="Path to the original model")
    parser.add_argument("--output_path", required=True, help="Path to save quantized model")
    parser.add_argument("--bits", type=int, choices=[4, 8], default=4, 
                       help="Quantization bits (4 or 8)")
    
    args = parser.parse_args()
    
    print("=== Aggressive Model Quantization ===")
    print(f"Input model: {args.model_path}")
    print(f"Output path: {args.output_path}")
    print(f"Quantization bits: {args.bits}")
    
    # Check original size
    print("\n--- Original Model Size ---")
    original_size = check_model_size(args.model_path)
    
    # Quantize
    print(f"\n--- Quantizing to {args.bits}-bit ---")
    success = aggressive_quantization(args.model_path, args.output_path, args.bits)
    
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
    else:
        print("✗ Quantization failed")

if __name__ == "__main__":
    main()
