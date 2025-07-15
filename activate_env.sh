#!/bin/bash
# Activation script for the virtual environment

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Cybersecurity Model Quantization Environment ===${NC}"
echo -e "${GREEN}Activating virtual environment...${NC}"

# Activate virtual environment
source venv/bin/activate

# Show activated environment
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo -e "${BLUE}Python version:${NC} $(python --version)"
echo -e "${BLUE}Pip version:${NC} $(pip --version)"

# Show available scripts
echo -e "\n${BLUE}Available scripts:${NC}"
echo "1. python quantize_model.py - Quantize your model"
echo "2. python test_quantized_models.py - Test quantized models"
echo "3. python quantized_model_api.py - Start the API server"
echo "4. python setup_quantization.py - Run full setup"

echo -e "\n${GREEN}Ready to start! Use 'deactivate' to exit the environment.${NC}"
