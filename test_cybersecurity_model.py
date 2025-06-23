#!/usr/bin/env python3
"""
Cybersecurity Model Tester
Tests the fine-tuned cybersecurity threat mitigation model
"""

import json
from unsloth import FastLanguageModel
from vllm import SamplingParams

def load_model(model_path="cybersecurity_threat_mitigation_lora"):
    """Load the trained cybersecurity model"""
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "meta-llama/Llama-3.2-3B-Instruct",
        max_seq_length = 5096,
        load_in_4bit = False,
        fast_inference = True,
        max_lora_rank = 8,
        gpu_memory_utilization = 0.6,
    )
    
    return model, tokenizer

def test_cybersecurity_model():
    """Test the model with various cybersecurity scenarios"""
    
    model, tokenizer = load_model()
    
    system_prompt = """You are a cybersecurity expert specializing in threat analysis and mitigation strategies. 

When analyzing a cybersecurity threat, provide your analysis in the following structured format:
1. <THREAT_ANALYSIS> Detailed threat analysis </THREAT_ANALYSIS>
2. <IMPACT_ASSESSMENT> Impact assessment </IMPACT_ASSESSMENT>
3. <MITIGATION_STRATEGY> Specific mitigation strategies </MITIGATION_STRATEGY>
4. <PREVENTION_MEASURES> Prevention measures </PREVENTION_MEASURES>

Focus on actionable recommendations and industry best practices."""

    # Test scenarios
    test_scenarios = [
        {
            "name": "Ransomware Attack",
            "scenario": """
Security Incident Alert:
We have detected a Ransomware attack targeting our Healthcare organization.

Incident Details:
- Attack Vector: Encrypts files, demands payment for decryption
- Vulnerability Exploited: Unpatched Software
- Attack Source: Cybercriminal Group
- Targeted Systems: Patient records, Medical devices, Administrative systems
- Tools/Techniques Used: CryptoLocker variants, Phishing emails

As our cybersecurity expert, please provide immediate response recommendations and long-term mitigation strategies.
"""
        },
        {
            "name": "APT Campaign",
            "scenario": """
Threat Intelligence Brief:
Recent threat intelligence indicates increased Advanced Persistent Threat activity targeting Financial Services sector.

Threat Characteristics:
- Common Attack Methods: Spear-phishing, Lateral movement, Data exfiltration
- Typical Vulnerabilities Exploited: Zero-day exploits, Social engineering
- Attacker Profile: Nation-state actor
- Target Assets: Customer financial data, Trading systems, Internal communications
- Known Tools: Custom malware, Living-off-the-land techniques

Develop preventive security measures and incident response procedures for this threat.
"""
        },
        {
            "name": "Supply Chain Attack",
            "scenario": """
Vulnerability Assessment Request:
Our security team has identified potential exposure to Supply Chain attacks in our Technology environment.

Risk Profile:
- Primary Vulnerability: Third-party software dependencies
- Potential Impact: Code injection, Data breach, System compromise
- Attack Complexity: High
- Threat Likelihood: Medium
- Compliance Concerns: SOX, PCI-DSS

Please provide a comprehensive security assessment and mitigation roadmap.
"""
        }
    ]
    
    sampling_params = SamplingParams(
        temperature = 0.7,
        top_p = 0.9,
        max_tokens = 1024,
    )
    
    for i, test in enumerate(test_scenarios):
        print(f"\n{'='*80}")
        print(f"TEST {i+1}: {test['name']}")
        print('='*80)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": test["scenario"]},
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True,
            tokenize = False,
        )
        
        print("INPUT SCENARIO:")
        print(test["scenario"])
        print("\n" + "-"*40 + " MODEL RESPONSE " + "-"*40)
        
        try:
            # Test with trained LoRA
            output = model.fast_generate(
                text,
                sampling_params = sampling_params,
                lora_request = model.load_lora("cybersecurity_threat_mitigation_lora"),
            )[0].outputs[0].text
            
            print(output)
            
            # Evaluate response quality
            evaluate_response(output, test['name'])
            
        except Exception as e:
            print(f"Error generating response: {e}")
            print("Testing without LoRA...")
            
            output = model.fast_generate(
                [text],
                sampling_params = sampling_params,
                lora_request = None,
            )[0].outputs[0].text
            
            print(output)

def evaluate_response(response, test_name):
    """Evaluate the quality of the cybersecurity response"""
    
    print("\n" + "-"*20 + " RESPONSE EVALUATION " + "-"*20)
    
    # Check for required sections
    required_sections = [
        "<THREAT_ANALYSIS>",
        "<IMPACT_ASSESSMENT>", 
        "<MITIGATION_STRATEGY>",
        "<PREVENTION_MEASURES>"
    ]
    
    section_scores = []
    for section in required_sections:
        if section in response and f"</{section.strip('<>')}>}" in response:
            section_scores.append(1)
            print(f"✓ {section}: Present and properly formatted")
        elif section in response:
            section_scores.append(0.5)
            print(f"~ {section}: Present but not properly closed")
        else:
            section_scores.append(0)
            print(f"✗ {section}: Missing")
    
    # Check for cybersecurity keywords
    cyber_keywords = [
        'mitigation', 'vulnerability', 'threat', 'security', 'attack',
        'prevention', 'monitoring', 'incident', 'risk', 'compliance',
        'authentication', 'encryption', 'firewall', 'patching', 'backup'
    ]
    
    keyword_count = sum(1 for keyword in cyber_keywords if keyword.lower() in response.lower())
    keyword_score = min(keyword_count / len(cyber_keywords), 1.0)
    
    # Check for actionable recommendations
    actionable_words = ['implement', 'deploy', 'configure', 'establish', 'conduct', 'review', 'update']
    actionable_count = sum(1 for word in actionable_words if word.lower() in response.lower())
    actionable_score = min(actionable_count / 5, 1.0)
    
    # Calculate overall score
    structure_score = sum(section_scores) / len(section_scores)
    overall_score = (structure_score * 0.4 + keyword_score * 0.3 + actionable_score * 0.3) * 100
    
    print(f"\nScoring for {test_name}:")
    print(f"Structure Score: {structure_score:.2f} ({sum(section_scores)}/{len(section_scores)} sections)")
    print(f"Keyword Coverage: {keyword_score:.2f} ({keyword_count}/{len(cyber_keywords)} keywords)")
    print(f"Actionable Content: {actionable_score:.2f} ({actionable_count}/5+ action words)")
    print(f"Overall Score: {overall_score:.1f}/100")

if __name__ == "__main__":
    print("Testing Cybersecurity Threat Mitigation Model")
    print("=" * 50)
    test_cybersecurity_model()
