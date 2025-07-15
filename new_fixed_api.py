from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import logging
import traceback
from datetime import datetime
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PORT = 5001
MODEL_VERSION = "1.0.0"

# Initialize model (you'll need to implement this based on your specific model)
class CybersecurityThreatAnalyzer:
    def __init__(self):
        self.model_loaded = False
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the cybersecurity threat analysis model"""
        try:
            # TODO: Implement your model loading logic here
            # This is a placeholder - replace with your actual model loading
            logger.info("Loading cybersecurity threat analysis model...")
            
            # Example model loading (replace with your actual implementation)
            # self.model = load_your_model_here()
            
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
            # TODO: Implement your threat analysis logic here
            # This is a placeholder response - replace with your actual model inference
            
            # Example analysis (replace with your actual implementation)
            analysis_result = {
                "threat_type": "Phishing",  # Replace with model prediction
                "severity": "High",  # Replace with model prediction
                "confidence": 0.85,  # Replace with model confidence
                "risk_score": 8.5,  # Replace with model risk score
                "description": threat_description,
                "analysis": {
                    "indicators": [
                        "Suspicious email patterns detected",
                        "Potential social engineering elements"
                    ],
                    "attack_vectors": ["Email", "Social Engineering"],
                    "potential_impact": "Data breach, credential theft"
                },
                "recommendations": [
                    "Implement email filtering",
                    "Conduct security awareness training",
                    "Enable multi-factor authentication"
                ],
                "mitigation_steps": [
                    "Block suspicious domains",
                    "Update security policies",
                    "Monitor for similar threats"
                ]
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing threat: {str(e)}")
            raise

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
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/quick-test', methods=['GET'])
def quick_test():
    """Quick test endpoint to verify the API is working"""
    try:
        if not analyzer.model_loaded:
            return jsonify({
                "success": False,
                "error": "Model not loaded",
                "timestamp": datetime.now().isoformat()
            }), 500
        
        # Test with a simple threat description
        test_threat = "Suspicious email received with urgent request for password reset"
        result = analyzer.analyze_threat(test_threat)
        
        return jsonify({
            "success": True,
            "message": "Quick test completed successfully",
            "test_input": test_threat,
            "test_result": result,
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Quick test error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "details": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/analyze-threat', methods=['POST'])
def analyze_threat():
    """Main endpoint for threat analysis"""
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data provided"
            }), 400
        
        threat_description = data.get('threat_description', '').strip()
        context = data.get('context', {})
        
        # Validate input
        if not threat_description:
            return jsonify({
                "success": False,
                "error": "threat_description is required and cannot be empty"
            }), 400
        
        # Check if model is loaded
        if not analyzer.model_loaded:
            return jsonify({
                "success": False,
                "error": "Model not loaded. Please check model API status."
            }), 500
        
        # Analyze the threat
        logger.info(f"Analyzing threat: {threat_description[:100]}...")
        result = analyzer.analyze_threat(threat_description, context)
        
        # Return successful response
        response = {
            "success": True,
            "analysis": result,
            "timestamp": datetime.now().isoformat(),
            "processed_by": "cybersecurity-threat-analyzer",
            "version": MODEL_VERSION
        }
        
        logger.info("Threat analysis completed successfully")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in analyze_threat: {str(e)}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            "success": False,
            "error": "Internal server error during threat analysis",
            "details": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    try:
        return jsonify({
            "success": True,
            "model_info": {
                "version": MODEL_VERSION,
                "loaded": analyzer.model_loaded,
                "service": "cybersecurity-threat-analyzer",
                "capabilities": [
                    "threat_classification",
                    "risk_assessment",
                    "recommendation_generation"
                ],
                "supported_threat_types": [
                    "Phishing",
                    "Ransomware",
                    "Malware",
                    "DDoS",
                    "Man-in-the-Middle",
                    "SQL Injection",
                    "Cross-Site Scripting (XSS)",
                    "Social Engineering",
                    "Brute Force",
                    "Zero-Day Exploit",
                    "Advanced Persistent Threat (APT)",
                    "Insider Threat"
                ]
            },
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found",
        "available_endpoints": [
            "/health",
            "/api/quick-test",
            "/api/analyze-threat",
            "/api/model-info"
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        "success": False,
        "error": "Internal server error",
        "timestamp": datetime.now().isoformat()
    }), 500

if __name__ == '__main__':
    logger.info(f"Starting Cybersecurity Threat Analyzer API on port {MODEL_PORT}")
    logger.info(f"Model loaded: {analyzer.model_loaded}")
    
    app.run(
        host='0.0.0.0',
        port=MODEL_PORT,
        debug=False,
        threaded=True
    )