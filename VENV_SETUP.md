# Virtual Environment Setup for Model Quantization

This guide will help you set up a virtual environment for quantizing your cybersecurity model.

## Quick Setup

1. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Or use the activation script:**
   ```bash
   ./activate_env.sh
   ```

3. **Test your model compatibility:**
   ```bash
   python test_model_compatibility.py
   ```

4. **Quantize your model:**
   ```bash
   python quantize_model.py --model_path /Users/abhigyasinha/Desktop/cybersecurity-threat-mitigation-instruct --output_path ./quantized_model_4bit --bits 4
   ```

## What's Installed

The virtual environment includes:

- **PyTorch** (2.7.1) - Deep learning framework
- **Transformers** (4.53.2) - Hugging Face transformers library
- **BitsAndBytes** (0.42.0) - Quantization library
- **Flask** (3.1.1) - Web framework for API
- **Accelerate** - Model loading optimizations
- **SafeTensors** - Secure tensor format

## File Structure

```
iic/
├── venv/                          # Virtual environment
├── activate_env.sh               # Environment activation script
├── requirements.txt              # Python dependencies
├── quantize_model.py            # Main quantization script
├── test_model_compatibility.py   # Test script
├── quantized_model_api.py       # API server
└── QUANTIZATION_GUIDE.md        # Detailed guide
```

## Usage

### 1. Activate Environment
```bash
source venv/bin/activate
```

### 2. Test Model
```bash
python test_model_compatibility.py
```

### 3. Quantize Model
```bash
# 4-bit quantization (recommended)
python quantize_model.py --model_path /path/to/your/model --output_path ./quantized_model_4bit --bits 4

# 8-bit quantization (larger but better quality)
python quantize_model.py --model_path /path/to/your/model --output_path ./quantized_model_8bit --bits 8
```

### 4. Test Quantized Model
```bash
python test_quantized_models.py
```

### 5. Start API Server
```bash
python quantized_model_api.py
```

## Expected Results

- **Original model**: ~6GB
- **4-bit quantized**: ~1.5GB (75% reduction)
- **8-bit quantized**: ~3GB (50% reduction)

## Troubleshooting

### BitsAndBytes Warning
```
The installed version of bitsandbytes was compiled without GPU support.
```
This is normal on macOS. The quantization will work in CPU mode.

### Memory Issues
If you encounter memory issues:
- Close other applications
- Use 4-bit quantization instead of 8-bit
- Consider using GGUF format for maximum compression

### Model Not Found
Make sure your model path is correct:
```bash
ls -la /Users/abhigyasinha/Desktop/cybersecurity-threat-mitigation-instruct/
```

## Deactivate Environment

When you're done:
```bash
deactivate
```

## Getting Help

1. Read the full guide: `cat QUANTIZATION_GUIDE.md`
2. Check the test results: `python test_model_compatibility.py`
3. Review the API documentation in `quantized_model_api.py`
