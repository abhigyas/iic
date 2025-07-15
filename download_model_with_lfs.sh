#!/bin/bash

# Script to download Hugging Face model using Git LFS
# This works even when Hugging Face Hub is blocked

echo "Downloading cybersecurity-threat-mitigation-instruct model with Git LFS..."

# Create directory for the model
MODEL_DIR="cybersecurity-threat-mitigation-instruct"
mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

# Clone the repository with Git LFS
git lfs clone https://huggingface.co/abhigyasinha/cybersecurity-threat-mitigation-instruct .

echo "Model downloaded successfully!"
echo "Files in the model directory:"
ls -la

echo ""
echo "Large files tracked by Git LFS:"
git lfs ls-files

echo ""
echo "To use the model, navigate to: $(pwd)"
