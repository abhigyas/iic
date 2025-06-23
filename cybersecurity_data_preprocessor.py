#!/usr/bin/env python3
"""
Cybersecurity Data Preprocessor
Converts raw cybersecurity incident data into structured training examples
for threat mitigation model training
"""

import json
import random
from typing import List, Dict, Any

def create_enhanced_training_data(input_file: str, output_file: str, max_examples: int = 2000):
    """
    Create enhanced training examples from cybersecurity data
    """
    
    with open(input_file, 'r') as f:
        cyber_data = json.load(f)
    
    training_examples = []
    
    # Template variations for different types of threat scenarios
    scenario_templates = [
        # Template 1: Incident Response Scenario
        """
Security Incident Alert:
We have detected a {attack_type} attack targeting our {target_industry} organization.

Incident Details:
- Attack Vector: {attack_method}
- Vulnerability Exploited: {vulnerability_type}
- Attack Source: {attack_source}
- Targeted Systems: {targeted_assets}
- Tools/Techniques Used: {tools_techniques}

As our cybersecurity expert, please provide immediate response recommendations and long-term mitigation strategies.
""",
        
        # Template 2: Vulnerability Assessment
        """
Vulnerability Assessment Request:
Our security team has identified potential exposure to {attack_type} attacks in our {target_industry} environment.

Risk Profile:
- Primary Vulnerability: {vulnerability_type}
- Potential Impact: {impact}
- Attack Complexity: {skill_level}
- Threat Likelihood: {likelihood}
- Compliance Concerns: {compliance_impact}

Please provide a comprehensive security assessment and mitigation roadmap.
""",
        
        # Template 3: Proactive Security Planning
        """
Threat Intelligence Brief:
Recent threat intelligence indicates increased {attack_type} activity targeting {target_industry} sector.

Threat Characteristics:
- Common Attack Methods: {attack_method}
- Typical Vulnerabilities Exploited: {vulnerability_type}
- Attacker Profile: {attack_source}
- Target Assets: {targeted_assets}
- Known Tools: {tools_techniques}

Develop preventive security measures and incident response procedures for this threat.
"""
    ]
    
    # Enhanced response templates with more detailed mitigation strategies
    def create_enhanced_response(item: Dict[str, Any], scenario_type: str) -> str:
        
        # Extract and clean data fields
        attack_type = item.get('Normalized_Attack_Type', item.get('Attack Type', 'Unknown'))
        category = item.get('Category', 'Unknown')
        subcategory = item.get('Subcategory', 'Unknown')
        likelihood = item.get('Likelihood', 'Medium')
        impact = item.get('Impact', 'Data compromise')
        mitigation = item.get('Mitigation/Prevention', 'Standard security protocols')
        
        if scenario_type == "incident_response":
            return f"""
<THREAT_ANALYSIS>
Attack Classification: {attack_type}
Threat Category: {category}
Specific Variant: {subcategory}
Threat Actor Profile: {item.get('Attack Source', 'Unknown')}
Attack Sophistication: {item.get('Skill Level of Attacker', 'Medium')}
Current Threat Level: {likelihood}
</THREAT_ANALYSIS>

<IMPACT_ASSESSMENT>
Immediate Impact: {impact}
Business Disruption: Potential {item.get('Incident Resolution Time (in Hours)', 'Unknown')} hours downtime
Financial Exposure: ${item.get('Financial Loss (in Million $)', 'Unknown')} million estimated loss
Affected Population: Up to {item.get('Number of Affected Users', 'Unknown')} users at risk
Regulatory Impact: {item.get('Compliance Impact', 'Multiple compliance frameworks may be affected')}
Reputational Risk: High potential for brand damage and customer trust erosion
</IMPACT_ASSESSMENT>

<MITIGATION_STRATEGY>
Immediate Actions (0-24 hours):
1. Activate incident response team and establish command center
2. Isolate affected systems to prevent lateral movement
3. Preserve forensic evidence and begin investigation
4. Implement emergency communication protocols
5. Deploy {item.get('Defense Mechanism Used', 'enhanced security controls')}

Short-term Actions (1-7 days):
1. {mitigation}
2. Patch identified vulnerabilities immediately
3. Reset credentials for potentially compromised accounts
4. Enhance monitoring for similar attack patterns
5. Conduct thorough security assessment of all systems

Long-term Actions (1-3 months):
1. Review and update security architecture
2. Implement additional security controls based on lessons learned
3. Conduct comprehensive security awareness training
4. Establish ongoing threat intelligence program
5. Regular penetration testing and vulnerability assessments
</MITIGATION_STRATEGY>

<PREVENTION_MEASURES>
Technical Controls:
- Network Segmentation: Implement micro-segmentation to limit attack spread
- Endpoint Protection: Deploy advanced EDR/XDR solutions
- Identity Management: Enforce multi-factor authentication and zero-trust principles
- Data Protection: Implement encryption at rest and in transit
- Monitoring: 24/7 SOC with behavioral analytics and threat hunting

Administrative Controls:
- Security Policies: Regular review and update of security procedures
- Training Programs: Quarterly cybersecurity awareness training
- Incident Response: Tested and updated incident response playbooks
- Vendor Management: Security assessments for all third-party providers
- Compliance: Regular audits and compliance monitoring

Physical Controls:
- Access Control: Biometric and card-based facility access
- Environmental: Secure data center with environmental monitoring
- Device Security: Asset tracking and mobile device management
- Visitor Management: Escort requirements and access logging
</PREVENTION_MEASURES>
"""
        
        elif scenario_type == "vulnerability_assessment":
            return f"""
<THREAT_ANALYSIS>
Vulnerability Type: {item.get('Security Vulnerability Type', 'Unknown')}
Attack Vector: {attack_type}
Exploit Complexity: {item.get('Skill Level of Attacker', 'Medium')}
Threat Landscape: {category} - {subcategory}
Industry Risk Profile: {item.get('Target Industry', 'Unknown')} sector commonly targeted
Attack Frequency: {item.get('Commonly Used in the Last 5 Years', 'Unknown')}
</THREAT_ANALYSIS>

<IMPACT_ASSESSMENT>
Risk Level: {likelihood} likelihood with {impact.lower()} impact
Business Impact: Operations disruption for {item.get('Duration', 'Unknown')} duration
Cost Analysis: Potential ${item.get('Financial Loss (in Million $)', 'Unknown')} million in losses
Data at Risk: {item.get('Targeted Assets', 'Unknown')}
Compliance Exposure: {item.get('Compliance Impact', 'Various regulations')}
Recovery Time: Estimated {item.get('Incident Resolution Time (in Hours)', 'Unknown')} hours
</IMPACT_ASSESSMENT>

<MITIGATION_STRATEGY>
Risk Reduction Approach:
1. Vulnerability Remediation: {mitigation}
2. Compensating Controls: Deploy {item.get('Defense Mechanism Used', 'additional security layers')}
3. Security Hardening: System and network configuration improvements
4. Access Controls: Implement principle of least privilege
5. Monitoring Enhancement: Deploy specific detection rules for this threat

Implementation Priority:
- Critical: Address vulnerabilities with active exploits (0-30 days)
- High: Implement primary prevention controls (30-60 days)
- Medium: Deploy enhanced monitoring and response (60-90 days)
- Low: Continuous improvement and optimization (ongoing)
</MITIGATION_STRATEGY>

<PREVENTION_MEASURES>
Proactive Security Measures:
- Threat Modeling: Regular assessment of attack vectors
- Security Testing: Automated vulnerability scanning and manual testing
- Code Security: Secure development lifecycle implementation
- Configuration Management: Baseline security configurations
- Patch Management: Automated patching with emergency procedures

Detection and Response:
- SIEM Integration: Correlation rules for {attack_type} detection
- Behavioral Analytics: User and entity behavior monitoring
- Threat Intelligence: Integration with external threat feeds
- Incident Response: Specialized playbooks for {category} attacks
- Forensics: Capability to investigate and analyze incidents
</PREVENTION_MEASURES>
"""
        
        else:  # proactive_security
            return f"""
<THREAT_ANALYSIS>
Emerging Threat: {attack_type}
Threat Campaign: Targeting {item.get('Target Industry', 'multiple industries')}
Attack Methodology: {item.get('Attack Method', 'Unknown')}
Adversary Capability: {item.get('Skill Level of Attacker', 'Medium')} sophistication level
Intelligence Confidence: High confidence based on {item.get('Data Breach Examples', 'recent incidents')}
Threat Persistence: {item.get('Commonly Used in the Last 5 Years', 'Unknown')} trending
</THREAT_ANALYSIS>

<IMPACT_ASSESSMENT>
Strategic Risk: High priority threat requiring immediate attention
Potential Impact: {impact} affecting {item.get('Target Audience', 'Unknown')}
Industry Exposure: {item.get('Target Industry', 'All sectors')} at elevated risk
Attack Timeline: {item.get('Duration', 'Unknown')} for full compromise
Financial Exposure: Industry average ${item.get('Financial Loss (in Million $)', 'Unknown')} million losses
Operational Impact: {item.get('Incident Resolution Time (in Hours)', 'Unknown')} hours average recovery time
</IMPACT_ASSESSMENT>

<MITIGATION_STRATEGY>
Defensive Strategy Framework:
1. Threat-Informed Defense: Align security controls with {attack_type} TTPs
2. Risk-Based Approach: Prioritize protection of {item.get('Targeted Assets', 'critical assets')}
3. Layered Security: Multiple defensive layers targeting {category} threats
4. Continuous Improvement: Regular assessment and enhancement of defenses

Strategic Implementation:
- Prevention: {mitigation}
- Detection: Enhanced monitoring for {item.get('Common Tools/Techniques', 'known attack tools')}
- Response: Specialized incident response procedures
- Recovery: Business continuity planning for {attack_type} scenarios
- Intelligence: Ongoing threat intelligence collection and analysis
</MITIGATION_STRATEGY>

<PREVENTION_MEASURES>
Strategic Security Program:
- Risk Management: Regular threat assessments and risk analysis
- Security Architecture: Design security into all systems and processes
- Cyber Resilience: Capability to maintain operations under attack
- Security Governance: Executive oversight and security metrics
- Stakeholder Engagement: Security awareness across all organizational levels

Tactical Security Controls:
- Prevention: {item.get('Defense Mechanism Used', 'Multi-layered security controls')}
- Detection: Signature and behavioral detection for {attack_type}
- Response: Automated and manual response capabilities
- Recovery: Tested backup and recovery procedures
- Intelligence: Threat intelligence integration and sharing
</PREVENTION_MEASURES>
"""
    
    # Generate training examples with different scenario types
    scenario_types = ["incident_response", "vulnerability_assessment", "proactive_security"]
    
    for i, item in enumerate(cyber_data):
        if len(training_examples) >= max_examples:
            break
            
        # Skip items with missing critical data
        if not item.get('Attack Type') or not item.get('Target Industry'):
            continue
        
        # Randomly select scenario type for variety
        scenario_type = random.choice(scenario_types)
        template = random.choice(scenario_templates)
        
        # Create threat scenario
        try:
            threat_scenario = template.format(
                attack_type=item.get('Attack Type', 'Unknown'),
                target_industry=item.get('Target Industry', 'Unknown'),
                attack_method=item.get('Attack Method', 'Unknown'),
                vulnerability_type=item.get('Security Vulnerability Type', 'Unknown'),
                attack_source=item.get('Attack Source', 'Unknown'),
                targeted_assets=item.get('Targeted Assets', 'Unknown'),
                tools_techniques=item.get('Common Tools/Techniques', 'Unknown'),
                impact=item.get('Impact', 'Unknown'),
                skill_level=item.get('Skill Level of Attacker', 'Medium'),
                likelihood=item.get('Likelihood', 'Medium'),
                compliance_impact=item.get('Compliance Impact', 'Various regulations')
            ).strip()
            
            mitigation_response = create_enhanced_response(item, scenario_type)
            
            training_examples.append({
                "id": i,
                "scenario_type": scenario_type,
                "threat_scenario": threat_scenario,
                "mitigation_response": mitigation_response.strip(),
                "attack_type": item.get('Attack Type'),
                "target_industry": item.get('Target Industry'),
                "metadata": {
                    "original_data": item
                }
            })
            
        except KeyError as e:
            print(f"Warning: Missing field {e} in record {i}, skipping...")
            continue
    
    # Save enhanced training data
    with open(output_file, 'w') as f:
        json.dump(training_examples, f, indent=2)
    
    print(f"Created {len(training_examples)} enhanced training examples")
    print(f"Scenario type distribution:")
    for scenario_type in scenario_types:
        count = sum(1 for ex in training_examples if ex['scenario_type'] == scenario_type)
        print(f"  {scenario_type}: {count} examples")
    
    return training_examples

if __name__ == "__main__":
    # Create enhanced training data
    enhanced_data = create_enhanced_training_data(
        input_file="/home/abhigya/iic/cyber_data.json",
        output_file="/home/abhigya/iic/cybersecurity_training_data.json",
        max_examples=1500
    )
    
    # Print sample example
    print("\n" + "="*60)
    print("SAMPLE TRAINING EXAMPLE")
    print("="*60)
    sample = enhanced_data[0]
    print(f"Scenario Type: {sample['scenario_type']}")
    print(f"\nThreat Scenario:\n{sample['threat_scenario'][:500]}...")
    print(f"\nMitigation Response:\n{sample['mitigation_response'][:500]}...")
    print("="*60)
