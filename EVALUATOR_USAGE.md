# Model Evaluator Usage Guide
# ==========================

This guide shows you how to use the comprehensive model evaluator to test your models.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r evaluator_requirements.txt
```

### 2. Quick Interactive Testing
```bash
# Simple interactive testing
python quick_tester.py /path/to/your/model

# Test with a dataset
python quick_tester.py /path/to/your/model test_data.json
```

### 3. Comprehensive Evaluation
```bash
# Full evaluation with test data
python model_evaluator.py /path/to/your/model --test-data test_data.json --visualize

# Interactive mode only
python model_evaluator.py /path/to/your/model --interactive

# Performance benchmark
python model_evaluator.py /path/to/your/model --benchmark

# Compare with baseline model
python model_evaluator.py /path/to/your/model --test-data test_data.json --compare /path/to/baseline/model
```

## Detailed Usage Examples

### Example 1: Evaluating Your Reasoning Model
```bash
# Test your reasoning model
python model_evaluator.py /home/abhigya/reasoning/reasoning \
    --test-data /home/abhigya/dataset/datasets_augmented_test.json \
    --benchmark \
    --visualize \
    --output reasoning_model_report.json
```

### Example 2: Comparing Models
```bash
# Compare your fine-tuned model with base model
python model_evaluator.py /home/abhigya/reasoning/reasoning \
    --test-data /home/abhigya/dataset/datasets_augmented_test.json \
    --compare unsloth/gemma-3-1b-it-bnb-4bit \
    --visualize
```

### Example 3: Interactive Testing Session
```bash
# Start interactive mode for manual testing
python model_evaluator.py /home/abhigya/reasoning/reasoning --interactive
```

### Example 4: Quick Validation
```bash
# Quick test with limited samples
python quick_tester.py /home/abhigya/reasoning/reasoning
```

## Using the Python API

### Basic Usage
```python
from model_evaluator import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator("/path/to/your/model")

# Load test data
evaluator.load_test_data("test_dataset.json")

# Run evaluations
qa_results = evaluator.evaluate_qa_accuracy("test_dataset")
quality_results = evaluator.evaluate_response_quality("test_dataset")
benchmark_results = evaluator.benchmark_performance()

# Generate report
evaluator.generate_report("my_evaluation_report.json")
evaluator.visualize_results()
```

### Advanced Usage
```python
# Custom generation parameters
evaluator = ModelEvaluator("/path/to/model")

# Test specific prompts
test_prompts = [
    "Explain quantum computing",
    "What is machine learning?",
    "How do neural networks work?"
]

# Custom response generation
for prompt in test_prompts:
    response = evaluator.generate_response(
        prompt, 
        max_new_tokens=200,
        temperature=0.8,
        top_p=0.9
    )
    print(f"Q: {prompt}")
    print(f"A: {response}\n")
```

## Test Data Format

Your test data should be in JSON format with questions and expected answers:

```json
[
    {
        "question": "What is the capital of France?",
        "answer": "Paris",
        "system": "Answer factual questions accurately."
    },
    {
        "instruction": "Explain photosynthesis",
        "output": "Photosynthesis is the process by which plants convert light energy into chemical energy..."
    }
]
```

Supported field names:
- `question` or `instruction`: The input question/prompt
- `answer` or `output`: Expected response
- `system`: Optional system prompt

## Understanding the Results

### Accuracy Metrics
- **Accuracy**: Percentage of questions answered correctly
- **Keyword Overlap**: Similarity based on word matching
- **Semantic Similarity**: Meaning-based comparison (if enabled)

### Quality Metrics
- **Coherence**: How well-structured and logical the response is
- **Relevance**: How well the response addresses the question
- **Completeness**: Whether the response fully answers the question
- **Fluency**: How natural and well-written the response is

### Performance Metrics
- **Response Time**: Average time to generate responses
- **Tokens per Second**: Generation speed
- **Token Count**: Length of generated responses

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Use CPU instead
   python model_evaluator.py /path/to/model --device cpu
   ```

2. **Model Loading Errors**
   ```bash
   # Specify custom tokenizer path
   python model_evaluator.py /path/to/model --tokenizer /path/to/tokenizer
   ```

3. **Slow Evaluation**
   ```bash
   # Use quick tester for faster results
   python quick_tester.py /path/to/model
   ```

### Performance Tips

1. **Use GPU**: Always use GPU for faster inference
2. **Batch Processing**: The evaluator automatically handles batching
3. **Sample Size**: Reduce sample size for faster testing
4. **Model Size**: Consider using quantized models for faster evaluation

## Output Files

The evaluator generates several output files:

- `evaluation_report_YYYYMMDD_HHMMSS.json`: Comprehensive results
- `evaluation_plots_YYYYMMDD_HHMMSS.png`: Visualization charts  
- `evaluation.log`: Detailed logging information

## Configuration

Modify `evaluator_config.json` to customize:
- Generation parameters
- Evaluation thresholds
- Test prompts
- Output formats
- Visualization settings

## Best Practices

1. **Diverse Test Data**: Use varied question types and domains
2. **Multiple Runs**: Run evaluations multiple times for consistency
3. **Baseline Comparison**: Always compare against a baseline model
4. **Interactive Testing**: Use interactive mode to explore edge cases
5. **Documentation**: Keep detailed records of evaluation results

## Example Workflow

1. **Initial Testing**: Use quick_tester.py for rapid validation
2. **Comprehensive Evaluation**: Run full evaluation with test dataset
3. **Baseline Comparison**: Compare against base model or previous version
4. **Interactive Exploration**: Use interactive mode to test edge cases
5. **Performance Analysis**: Review metrics and visualizations
6. **Iterative Improvement**: Use results to guide model improvements

This evaluator will help you thoroughly test your model and understand its capabilities and limitations!
