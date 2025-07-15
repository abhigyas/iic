#!/usr/bin/env python3
"""
Quick test to verify your model can be quantized
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def test_model_loading():
    """Test loading the original model"""
    model_path = "/Users/abhigyasinha/Desktop/cybersecurity-threat-mitigation-instruct"
    
    print(f"Testing model loading from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"✗ Model not found at {model_path}")
        return False
    
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✓ Tokenizer loaded successfully")
        
        print("Loading model configuration...")
        # First, just load the config to check it's valid
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path)
        print(f"✓ Model config loaded: {config.model_type}")
        
        print("Testing quantization configuration...")
        # Test BitsAndBytes config
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        print("✓ Quantization config created successfully")
        
        # Test loading with quantization (this will be slow but proves it works)
        print("Testing model loading with quantization (this may take a while)...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            print("✓ Model loaded with quantization successfully!")
            
            # Test a simple inference
            print("Testing simple inference...")
            test_input = "What is cybersecurity?"
            inputs = tokenizer(test_input, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=50,
                    num_return_sequences=1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"✓ Inference test successful!")
            print(f"Response: {response}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error during quantized model loading: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Error during model testing: {e}")
        return False

def main():
    print("=== Model Quantization Test ===\n")
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Please activate the virtual environment first:")
        print("   source venv/bin/activate")
        sys.exit(1)
    
    print("✓ Virtual environment is activated")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    if test_model_loading():
        print("\n✓ Your model is ready for quantization!")
        print("Next steps:")
        print("1. Run: python quantize_model.py --model_path /Users/abhigyasinha/Desktop/cybersecurity-threat-mitigation-instruct --output_path ./quantized_model_4bit --bits 4")
        print("2. Test: python test_quantized_models.py")
        print("3. Start API: python quantized_model_api.py")
    else:
        print("\n✗ There were issues with your model. Please check the errors above.")

if __name__ == "__main__":
    main()
