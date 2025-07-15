from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import logging
import traceback
from datetime import datetime
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PORT = 5001
MODEL_VERSION = "1.0.0"
MODEL_PATH = "./quantized_model_4bit"  # Path to your quantized model

# Initialize model
class CybersecurityThreatAnalyzer:
    def __init__(self):
        self.model_loaded = False
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load the cybersecurity threat analysis model"""
        try:
            logger.info("Loading cybersecurity threat analysis model...")
            
            # Check if quantized model exists
            if not os.path.exists(MODEL_PATH):
                logger.error(f"Model not found at {MODEL_PATH}")
                logger.info("Please run quantization first using: python quantize_model.py")
                self.model_loaded = False
                return
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            
            # Load quantized model
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            self.model_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model_loaded = False
    
    def analyze_threat(self, threat_description, context=None):
        """Analyze a cybersecurity threat and return classification and recommendations"""
        if not self.model_loaded:
            raise Exception("Model not loaded")
        
        try:
            # Create a prompt for the cybersecurity model
            prompt = f"""
            Analyze the following cybersecurity threat and provide a detailed analysis:
            
            Threat Description: {threat_description}
            {f"Context: {context}" if context else ""}
            
            Please provide:
            1. Threat type classification
            2. Severity level (Low/Medium/High/Critical)
            3. Risk score (1-10)
            4. Potential attack vectors
            5. Recommended mitigation steps
            
            Analysis:
            """
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=1024,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the analysis part (everything after "Analysis:")
            analysis_text = response.split("Analysis:")[-1].strip()
            
            # Parse the response into structured format
            analysis_result = self._parse_analysis(analysis_text, threat_description)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing threat: {str(e)}")
            raise
    
    def _parse_analysis(self, analysis_text, threat_description):
        """Parse the model output into structured format"""
        try:
            # This is a simplified parser - you might want to make it more robust
            analysis_result = {
                "threat_type": "Unknown",
                "severity": "Medium",
                "confidence": 0.7,
                "risk_score": 5.0,
                "description": threat_description,
                "analysis": {
                    "full_analysis": analysis_text,
                    "indicators": [],
                    "attack_vectors": [],
                    "potential_impact": "To be determined"
                },
                "recommendations": [],
                "mitigation_steps": []
            }
            
            # Extract key information (basic parsing)
            lines = analysis_text.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Look for severity indicators
                if any(word in line.lower() for word in ['critical', 'high', 'medium', 'low']):
                    if 'critical' in line.lower():
                        analysis_result["severity"] = "Critical"
                        analysis_result["risk_score"] = 9.0
                    elif 'high' in line.lower():
                        analysis_result["severity"] = "High"
                        analysis_result["risk_score"] = 7.5
                    elif 'medium' in line.lower():
                        analysis_result["severity"] = "Medium"
                        analysis_result["risk_score"] = 5.0
                    elif 'low' in line.lower():
                        analysis_result["severity"] = "Low"
                        analysis_result["risk_score"] = 2.5
                
                # Look for threat types
                threat_types = ['phishing', 'malware', 'ddos', 'ransomware', 'social engineering', 'insider threat']
                for threat_type in threat_types:
                    if threat_type in line.lower():
                        analysis_result["threat_type"] = threat_type.title()
                        break
                
                # Look for recommendations
                if any(word in line.lower() for word in ['recommend', 'should', 'implement', 'enable']):
                    analysis_result["recommendations"].append(line)
                
                # Look for mitigation steps
                if any(word in line.lower() for word in ['mitigate', 'prevent', 'block', 'update']):
                    analysis_result["mitigation_steps"].append(line)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error parsing analysis: {str(e)}")
            # Return basic structure if parsing fails
            return {
                "threat_type": "Unknown",
                "severity": "Medium",
                "confidence": 0.5,
                "risk_score": 5.0,
                "description": threat_description,
                "analysis": {
                    "full_analysis": analysis_text,
                    "indicators": [],
                    "attack_vectors": [],
                    "potential_impact": "Analysis parsing failed"
                },
                "recommendations": ["Consult security expert"],
                "mitigation_steps": ["Manual review required"]
            }

# Initialize the analyzer
analyzer = CybersecurityThreatAnalyzer()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_version": MODEL_VERSION,
            "model_loaded": analyzer.model_loaded,
            "service": "cybersecurity-threat-analyzer"
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }), 500

@app.route('/analyze', methods=['POST'])
def analyze_threat():
    """Analyze a cybersecurity threat"""
    try:
        data = request.get_json()
        
        if not data or 'threat_description' not in data:
            return jsonify({
                "error": "Missing 'threat_description' in request body"
            }), 400
        
        threat_description = data['threat_description']
        context = data.get('context', None)
        
        # Analyze the threat
        result = analyzer.analyze_threat(threat_description, context)
        
        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "result": result
        }), 200
        
    except Exception as e:
        logger.error(f"Error in analyze_threat: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }), 500

@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    """Analyze multiple threats at once"""
    try:
        data = request.get_json()
        
        if not data or 'threats' not in data:
            return jsonify({
                "error": "Missing 'threats' array in request body"
            }), 400
        
        threats = data['threats']
        results = []
        
        for threat in threats:
            if isinstance(threat, dict) and 'threat_description' in threat:
                result = analyzer.analyze_threat(
                    threat['threat_description'],
                    threat.get('context', None)
                )
                results.append(result)
            else:
                results.append({
                    "error": "Invalid threat format",
                    "threat": threat
                })
        
        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "results": results
        }), 200
        
    except Exception as e:
        logger.error(f"Error in batch_analyze: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    try:
        if not analyzer.model_loaded:
            return jsonify({
                "error": "Model not loaded"
            }), 503
        
        # Get model size
        model_size_mb = 0
        if os.path.exists(MODEL_PATH):
            for root, dirs, files in os.walk(MODEL_PATH):
                for file in files:
                    file_path = os.path.join(root, file)
                    model_size_mb += os.path.getsize(file_path)
            model_size_mb = model_size_mb / (1024 * 1024)
        
        return jsonify({
            "status": "success",
            "model_info": {
                "model_path": MODEL_PATH,
                "model_loaded": analyzer.model_loaded,
                "model_size_mb": round(model_size_mb, 2),
                "version": MODEL_VERSION,
                "capabilities": [
                    "Threat classification",
                    "Severity assessment",
                    "Risk scoring",
                    "Mitigation recommendations"
                ]
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    logger.info(f"Starting Cybersecurity Threat Analyzer API on port {MODEL_PORT}")
    app.run(host='0.0.0.0', port=MODEL_PORT, debug=True)
