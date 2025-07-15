#!/usr/bin/env python3
"""
Quick setup script for model quantization
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("Installing quantization requirements...")
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Virtual environment not detected. Please activate it first:")
        print("   source venv/bin/activate")
        return False
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "-r", "requirements.txt"
        ], check=True)
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False

def quantize_model():
    """Run the quantization process"""
    print("Starting model quantization...")
    
    model_path = "/Users/abhigyasinha/Desktop/cybersecurity-threat-mitigation-instruct"
    output_path = "./quantized_model_4bit"
    
    if not os.path.exists(model_path):
        print(f"✗ Model not found at {model_path}")
        return False
    
    try:
        subprocess.run([
            sys.executable, "quantize_model.py",
            "--model_path", model_path,
            "--output_path", output_path,
            "--bits", "4"
        ], check=True)
        print("✓ Model quantization completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during quantization: {e}")
        return False

def check_model_size():
    """Check the size of the quantized model"""
    output_path = "./quantized_model_4bit"
    
    if not os.path.exists(output_path):
        print("✗ Quantized model not found")
        return False
    
    total_size = 0
    for root, dirs, files in os.walk(output_path):
        for file in files:
            file_path = os.path.join(root, file)
            total_size += os.path.getsize(file_path)
    
    size_mb = total_size / (1024 * 1024)
    print(f"✓ Quantized model size: {size_mb:.2f} MB")
    
    if size_mb <= 100:
        print("✓ Model is small enough for GitHub!")
    else:
        print("⚠️  Model is still larger than 100MB")
        print("Consider using GGUF format or external hosting")
    
    return True

def main():
    print("=== Cybersecurity Model Quantization Setup ===\n")
    
    steps = [
        ("Installing requirements", install_requirements),
        ("Quantizing model", quantize_model),
        ("Checking model size", check_model_size)
    ]
    
    for step_name, step_func in steps:
        print(f"\n--- {step_name} ---")
        if not step_func():
            print(f"✗ Failed at step: {step_name}")
            sys.exit(1)
    
    print("\n=== Setup Complete ===")
    print("Next steps:")
    print("1. Test your quantized model: python test_quantized_models.py")
    print("2. Start the API: python quantized_model_api.py")
    print("3. Read the full guide: cat QUANTIZATION_GUIDE.md")

if __name__ == "__main__":
    main()
