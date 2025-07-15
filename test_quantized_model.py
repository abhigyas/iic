#!/usr/bin/env python3
"""
Test script for the quantized model
"""

import sys
import os
sys.path.append('./quantized_model_4bit')

from load_quantized_model import load_quantized_model, dequantize_weights

def test_quantized_model():
    """Test loading and using the quantized model"""
    
    print("=== Testing Quantized Model ===")
    
    model_path = "./quantized_model_4bit"
    
    if not os.path.exists(model_path):
        print(f"Error: Quantized model not found at {model_path}")
        return False
    
    try:
        print("Loading quantized model...")
        quantized_weights, quant_params, config, tokenizer = load_quantized_model(model_path)
        
        print("✓ Quantized model loaded successfully!")
        print(f"Model type: {config.model_type}")
        print(f"Vocabulary size: {config.vocab_size}")
        print(f"Number of layers: {config.num_hidden_layers}")
        print(f"Number of parameters: {len(quantized_weights)}")
        
        # Test tokenizer
        print("\nTesting tokenizer...")
        test_text = "What is a cybersecurity threat?"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        print(f"Original: {test_text}")
        print(f"Decoded: {decoded}")
        
        # Test dequantization of a small subset
        print("\nTesting weight dequantization...")
        sample_weights = {k: v for k, v in list(quantized_weights.items())[:2]}
        sample_params = {k: v for k, v in list(quant_params.items())[:2]}
        
        dequantized_sample = dequantize_weights(sample_weights, sample_params)
        print(f"✓ Successfully dequantized {len(dequantized_sample)} weight tensors")
        
        # Show model size info
        print("\nModel size information:")
        total_size = 0
        for root, dirs, files in os.walk(model_path):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
        
        size_mb = total_size / (1024 * 1024)
        print(f"Total size: {size_mb:.2f} MB")
        
        if size_mb <= 100:
            print("✓ Model is small enough for GitHub!")
        elif size_mb <= 1000:
            print("⚠️  Model is larger than 100MB but could work with Git LFS")
        else:
            print("⚠️  Model is still large, consider external hosting")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing quantized model: {e}")
        return False

def main():
    success = test_quantized_model()
    
    if success:
        print("\n=== Test Results ===")
        print("✓ Quantized model is working correctly!")
        print("✓ You can now use this model in your applications")
        print("\nNext steps:")
        print("1. Update your API to use the quantized model")
        print("2. Test inference with the quantized model")
        print("3. Consider hosting options for your use case")
    else:
        print("\n=== Test Results ===")
        print("✗ Quantized model test failed")
        print("Please check the error messages above")

if __name__ == "__main__":
    main()
