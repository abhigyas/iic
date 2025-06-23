#!/usr/bin/env python3
"""
Cybersecurity Threat Mitigation Training Script
Fine-tunes Llama 3.2 3B for cybersecurity threat analysis and mitigation strategies
"""

from unsloth import FastLanguageModel
import torch
import json
import re
from datasets import Dataset, load_dataset
from trl import GRPOConfig, GRPOTrainer

# Model Configuration
max_seq_length = 2048
lora_rank = 8

# Load the model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/Llama-3.2-3B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = False,
    fast_inference = True,
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# Data Preparation for Cybersecurity
def load_enhanced_training_data(data_file):
    """Load preprocessed cybersecurity training data"""
    with open(data_file, 'r') as f:
        training_data = json.load(f)
    
    return training_data

# First, create enhanced training data if it doesn't exist
import subprocess
import os

enhanced_data_file = "/home/abhigya/iic/cybersecurity_training_data.json"
if not os.path.exists(enhanced_data_file):
    print("Creating enhanced training data...")
    subprocess.run(["python3", "/home/abhigya/iic/cybersecurity_data_preprocessor.py"])

# Load and prepare cybersecurity data
print("Loading enhanced cybersecurity training data...")
training_data = load_enhanced_training_data(enhanced_data_file)

# Convert to dataset format
clean_dataset = []
for data in training_data[:1200]:  # Use more examples with enhanced data
    clean_dataset.append({
        "question": data["threat_scenario"],
        "answer": data["mitigation_response"]
    })

dataset = Dataset.from_list(clean_dataset)

# Cybersecurity-specific response format markers
analysis_start = "<THREAT_ANALYSIS>"
analysis_end = "</THREAT_ANALYSIS>"
impact_start = "<IMPACT_ASSESSMENT>"
impact_end = "</IMPACT_ASSESSMENT>"
mitigation_start = "<MITIGATION_STRATEGY>"
mitigation_end = "</MITIGATION_STRATEGY>"
prevention_start = "<PREVENTION_MEASURES>"
prevention_end = "</PREVENTION_MEASURES>"

# Custom system prompt for cybersecurity
system_prompt = f"""You are a cybersecurity expert specializing in threat analysis and mitigation strategies. 

When analyzing a cybersecurity threat, provide your analysis in the following structured format:
1. {analysis_start} Detailed threat analysis {analysis_end}
2. {impact_start} Impact assessment {impact_end}
3. {mitigation_start} Specific mitigation strategies {mitigation_end}
4. {prevention_start} Prevention measures {prevention_end}

Focus on actionable recommendations and industry best practices."""

# Map dataset with new format
dataset = dataset.map(lambda x: {
    "prompt": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": x["question"]},
    ],
    "answer": x["answer"],
})

print(f"Dataset prepared with {len(dataset)} examples")
print("Sample question:", dataset[0]["prompt"][1]["content"][:200], "...")
print("Sample answer:", dataset[0]["answer"][:200], "...")

# Regex patterns for cybersecurity response format
cyber_format = re.compile(
    rf"^[\s]{{0,}}"\
    rf"{analysis_start}.+?{analysis_end}.*?"\
    rf"{impact_start}.+?{impact_end}.*?"\
    rf"{mitigation_start}.+?{mitigation_end}.*?"\
    rf"{prevention_start}.+?{prevention_end}"\
    rf"[\s]{{0,}}$",
    flags = re.MULTILINE | re.DOTALL
)

# Reward function for complete cybersecurity format
def match_cyber_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        if cyber_format.search(response) is not None:
            score += 5.0  # Higher reward for complete format
        scores.append(score)
    return scores

# Reward function for partial cybersecurity format
def match_cyber_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        
        # Check for each required section
        score += 1.0 if analysis_start in response and analysis_end in response else -0.5
        score += 1.0 if impact_start in response and impact_end in response else -0.5
        score += 1.0 if mitigation_start in response and mitigation_end in response else -0.5
        score += 1.0 if prevention_start in response and prevention_end in response else -0.5
        
        scores.append(score)
    return scores

# Reward function for cybersecurity content quality
def check_cyber_content(prompts, completions, answer, **kwargs):
    scores = []
    
    for completion, true_answer in zip(completions, answer):
        score = 0
        response = completion[0]["content"].lower()
        true_answer_lower = true_answer[0].lower() if isinstance(true_answer, list) else true_answer.lower()
        
        # Reward for cybersecurity keywords
        cyber_keywords = [
            'mitigation', 'vulnerability', 'threat', 'security', 'attack',
            'prevention', 'monitoring', 'incident', 'risk', 'compliance',
            'authentication', 'encryption', 'firewall', 'patching', 'backup'
        ]
        
        keyword_matches = sum(1 for keyword in cyber_keywords if keyword in response)
        score += keyword_matches * 0.2
        
        # Reward for structured thinking
        if 'immediate' in response and 'long-term' in response:
            score += 1.0
        
        # Reward for specific technical controls mentioned
        technical_controls = ['network segmentation', 'endpoint protection', 'multi-factor', 'access control']
        tech_matches = sum(1 for control in technical_controls if control in response)
        score += tech_matches * 0.5
        
        scores.append(score)
    
    return scores

# Training progress tracking
global PRINTED_TIMES, PRINT_EVERY_STEPS
PRINTED_TIMES = 0
PRINT_EVERY_STEPS = 10

def print_cyber_progress(prompts, completions, answer, **kwargs):
    global PRINTED_TIMES, PRINT_EVERY_STEPS
    
    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        question = prompts[0][-1]["content"][:300]
        response = completions[0][0]["content"][:500]
        
        print('='*50)
        print(f"Training Step {PRINTED_TIMES}")
        print(f"Question: {question}...")
        print(f"Response: {response}...")
        print('='*50)
    
    PRINTED_TIMES += 1
    return [0.0] * len(completions)  # Neutral score, just for monitoring

# Calculate maximum prompt length
max_prompt_length = max(dataset.map(
    lambda x: {"tokens": tokenizer.apply_chat_template(x["prompt"], add_generation_prompt=True, tokenize=True)},
    batched=True,
).map(lambda x: {"length": len(x["tokens"])})["length"]) + 10

print(f"Maximum prompt length: {max_prompt_length}")

# Training configuration
training_args = GRPOConfig(
    learning_rate = 1e-5,  # Lower learning rate for cybersecurity domain
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 25,
    per_device_train_batch_size = 4,  # Smaller batch size for complex responses
    gradient_accumulation_steps = 4,
    num_generations = 2,
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length - max_prompt_length,
    # num_train_epochs = 1,  # More epochs for domain expertise
    max_steps = 500,
    save_steps = 100,
    max_grad_norm = 1.0,
    report_to = "none",
    output_dir = "cybersecurity_outputs",
)

# Initialize trainer with cybersecurity-specific reward functions
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        match_cyber_format_exactly,
        match_cyber_format_approximately,
        check_cyber_content,
        print_cyber_progress,
    ],
    args = training_args,
    train_dataset = dataset,
)

print("Starting cybersecurity threat mitigation training...")
trainer.train()

# Save the trained model
model.save_lora("cybersecurity_threat_mitigation_lora")
print("Model saved to: cybersecurity_threat_mitigation_lora")

# Test the trained model
print("\n" + "="*60)
print("TESTING TRAINED CYBERSECURITY MODEL")
print("="*60)

test_scenario = """
Threat Analysis Request:
- Attack Type: SQL Injection
- Target Industry: Healthcare
- Attack Source: External Attacker
- Security Vulnerability: Unvalidated Input
- Targeted Assets: Patient Database
- Attack Method: Malicious SQL queries through web forms
- Common Tools/Techniques: SQLMap, Manual injection techniques

Please provide a comprehensive threat analysis and mitigation strategy for this cybersecurity incident.
"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": test_scenario},
]

text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    tokenize = False,
)

from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature = 0.7,
    top_p = 0.9,
    max_tokens = 1024,
)

output = model.fast_generate(
    text,
    sampling_params = sampling_params,
    lora_request = model.load_lora("cybersecurity_threat_mitigation_lora"),
)[0].outputs[0].text

print("Test Scenario:")
print(test_scenario)
print("\nModel Response:")
print(output)

# Save final model versions
if True:
    model.save_pretrained("cybersecurity-threat-mitigation")
    tokenizer.save_pretrained("cybersecurity-threat-mitigation")
    
if True: 
    model.save_pretrained_merged(
        "cybersecurity-threat-mitigation-16bit", 
        tokenizer, 
        save_method = "merged_16bit"
    )

print("\nTraining completed! Model saved for cybersecurity threat mitigation.")
