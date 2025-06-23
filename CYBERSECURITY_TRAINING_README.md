# Cybersecurity Threat Mitigation Model Training

## Overview
This project fine-tunes a Llama 3.2 3B model specifically for cybersecurity threat analysis and mitigation strategies using GRPO (Generalized Reinforcement Learning from Policy Optimization).

## Key Changes Made

### 1. Data Transformation
**Original Code**: Used simple prompt-response pairs from mathematical reasoning data
**New Code**: Created structured cybersecurity scenario-mitigation pairs with three scenario types:
- **Incident Response**: Real-time security incident analysis
- **Vulnerability Assessment**: Proactive risk assessment 
- **Proactive Security**: Strategic threat intelligence planning

### 2. Response Format
**Original Format**: `<start_working_out>` reasoning `<end_working_out>` `<SOLUTION>` answer `</SOLUTION>`
**New Format**: 
```
<THREAT_ANALYSIS> threat classification </THREAT_ANALYSIS>
<IMPACT_ASSESSMENT> business impact analysis </IMPACT_ASSESSMENT>
<MITIGATION_STRATEGY> specific countermeasures </MITIGATION_STRATEGY>
<PREVENTION_MEASURES> long-term security controls </PREVENTION_MEASURES>
```

### 3. System Prompt
**Original**: Generic mathematical reasoning prompt
**New**: Cybersecurity expert persona with structured analysis requirements

### 4. Reward Functions
**Original**: Numerical answer matching and format compliance
**New**: Cybersecurity-specific rewards:
- **Format compliance**: Proper security analysis structure
- **Content quality**: Cybersecurity terminology and concepts
- **Actionable recommendations**: Practical mitigation steps

### 5. Training Parameters
- **Learning rate**: Reduced to 1e-5 for domain-specific fine-tuning
- **Batch size**: Reduced to 4 for complex cybersecurity responses
- **Epochs**: Increased to 2 for better domain adaptation
- **Steps**: Increased to 1000 for comprehensive training

## File Structure

```
/home/abhigya/iic/
├── cybersecurity_data_preprocessor.py     # Converts raw data to training format
├── cybersecurity_threat_mitigation_trainer.py  # Main training script
├── test_cybersecurity_model.py           # Model evaluation script
├── cyber_data.json                       # Original cybersecurity data
└── cybersecurity_training_data.json      # Enhanced training examples
```

## How to Use

### 1. Prepare Training Data
```bash
python3 cybersecurity_data_preprocessor.py
```
This creates 1,500 enhanced training examples with realistic cybersecurity scenarios.

### 2. Train the Model
```bash
python3 cybersecurity_threat_mitigation_trainer.py
```
This will:
- Load the Llama 3.2 3B model
- Apply LoRA fine-tuning with cybersecurity-specific rewards
- Save the trained model as `cybersecurity_threat_mitigation_lora`

### 3. Test the Model
```bash
python3 test_cybersecurity_model.py
```
This evaluates the model on various threat scenarios and provides scoring.

## Training Data Examples

### Incident Response Scenario
```
Security Incident Alert:
We have detected a Ransomware attack targeting our Healthcare organization.

Incident Details:
- Attack Vector: Encrypts files, demands payment for decryption
- Vulnerability Exploited: Unpatched Software
- Attack Source: Cybercriminal Group
- Tools/Techniques Used: CryptoLocker variants
```

### Expected Response Format
```
<THREAT_ANALYSIS>
Attack Classification: Ransomware
Threat Category: Malware
Specific Variant: Encrypting Ransomware
Threat Actor Profile: Cybercriminal Group
</THREAT_ANALYSIS>

<IMPACT_ASSESSMENT>
Immediate Impact: Data loss, Financial loss, Downtime
Business Disruption: Potential 71 hours downtime
Financial Exposure: $62.19 million estimated loss
</IMPACT_ASSESSMENT>

<MITIGATION_STRATEGY>
Immediate Actions (0-24 hours):
1. Activate incident response team
2. Isolate affected systems
3. Preserve forensic evidence
...
</MITIGATION_STRATEGY>

<PREVENTION_MEASURES>
Technical Controls:
- Network Segmentation
- Endpoint Protection
- Regular Backups
...
</PREVENTION_MEASURES>
```

## Model Capabilities

After training, the model can:

1. **Analyze Threats**: Classify attack types, assess threat actors, evaluate sophistication
2. **Assess Impact**: Calculate business impact, financial exposure, regulatory implications
3. **Recommend Mitigations**: Provide immediate, short-term, and long-term response strategies
4. **Suggest Prevention**: Offer technical, administrative, and physical security controls

## Evaluation Metrics

The test script evaluates responses based on:
- **Structure Score**: Presence of required analysis sections (40%)
- **Keyword Coverage**: Use of cybersecurity terminology (30%)
- **Actionable Content**: Practical recommendations (30%)

## Benefits of This Approach

1. **Domain Expertise**: Model learns cybersecurity-specific language and concepts
2. **Structured Analysis**: Consistent, professional security assessment format
3. **Actionable Output**: Practical mitigation strategies, not just theoretical knowledge
4. **Scalable Training**: Easy to add new threat scenarios and attack types
5. **Quality Assurance**: Built-in evaluation metrics for response quality

## Next Steps

1. **Expand Dataset**: Add more attack types (IoT, Cloud, AI/ML threats)
2. **Industry Specialization**: Create industry-specific training data
3. **Integration**: Connect with SIEM/SOAR platforms for real-time analysis
4. **Continuous Learning**: Regular retraining with latest threat intelligence
5. **Validation**: Test against real incident response scenarios

This training approach transforms a general-purpose language model into a specialized cybersecurity expert capable of providing structured threat analysis and actionable mitigation strategies.
