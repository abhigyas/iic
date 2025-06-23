#!/usr/bin/env python3


import json
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuickModelTester:
    """Simplified model tester for rapid evaluation."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        logger.info("Model loaded successfully!")
    
    def test_question(self, question: str, max_tokens: int = 256) -> dict:
        """Test a single question."""
        prompt = f"Question: {question}\nAnswer:"
        
        start_time = time.time()
        response = self.pipeline(
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            return_full_text=False
        )[0]['generated_text']
        
        response_time = time.time() - start_time
        
        return {
            'question': question,
            'answer': response.strip(),
            'response_time': response_time,
            'tokens_generated': len(self.tokenizer.encode(response))
        }
    
    def quick_benchmark(self, test_questions: list = None) -> dict:
        """Run a quick benchmark."""
        if test_questions is None:
            test_questions = [
                "What is artificial intelligence?",
                "How does machine learning work?",
                "Explain deep learning in simple terms.",
                "What are the benefits of renewable energy?",
                "How do computers process information?"
            ]
        
        results = []
        total_time = 0
        
        print("\nRunning Quick Benchmark...")
        print("-" * 50)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. Testing: {question}")
            result = self.test_question(question)
            results.append(result)
            total_time += result['response_time']
            
            print(f"   Answer: {result['answer'][:100]}...")
            print(f"   Time: {result['response_time']:.2f}s")
        
        avg_time = total_time / len(test_questions)
        avg_tokens = sum(r['tokens_generated'] for r in results) / len(results)
        
        summary = {
            'num_questions': len(test_questions),
            'total_time': total_time,
            'avg_response_time': avg_time,
            'avg_tokens_generated': avg_tokens,
            'tokens_per_second': avg_tokens / avg_time,
            'results': results
        }
        
        print("\nBenchmark Summary:")
        print("-" * 50)
        print(f"Questions tested: {summary['num_questions']}")
        print(f"Average response time: {summary['avg_response_time']:.2f}s")
        print(f"Average tokens generated: {summary['avg_tokens_generated']:.1f}")
        print(f"Tokens per second: {summary['tokens_per_second']:.1f}")
        
        return summary
    
    def interactive_chat(self):
        """Interactive chat mode."""
        print("\nInteractive Chat Mode")
        print("Type 'quit' to exit, 'benchmark' to run quick test")
        print("-" * 50)
        
        while True:
            try:
                question = input("\nYou: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                elif question.lower() == 'benchmark':
                    self.quick_benchmark()
                    continue
                elif not question:
                    continue
                
                print("AI: ", end="", flush=True)
                result = self.test_question(question)
                print(result['answer'])
                print(f"    (Time: {result['response_time']:.2f}s)")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def test_dataset(self, dataset_path: str, max_samples: int = 10) -> dict:
        """Test on a dataset."""
        with open(dataset_path, 'r') as f:
            if dataset_path.endswith('.jsonl'):
                data = [json.loads(line) for line in f]
            else:
                data = json.load(f)
        
        if max_samples:
            data = data[:max_samples]
        
        print(f"\nTesting on dataset: {Path(dataset_path).name}")
        print(f"Testing {len(data)} samples...")
        
        # Create output filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"test_results_{timestamp}.json"
        
        results = []
        correct = 0
        
        for i, sample in enumerate(data):
            # Support multiple field name formats
            question = sample.get('threat_scenario', 
                                sample.get('question', 
                                         sample.get('instruction', '')))
            expected = sample.get('mitigation_response',
                                sample.get('answer', 
                                         sample.get('output', '')))
            
            if not question:
                print(f"  Skipping sample {i+1}: No question field found")
                continue
            
            if not expected:
                print(f"  Skipping sample {i+1}: No expected answer field found")
                continue
            
            print(f"\nTesting sample {i+1}/{len(data)}")
            print(f"Question: {question[:100]}...")
            
            # Use more tokens for complex cybersecurity responses
            result = self.test_question(question, max_tokens=512)
            actual = result['answer']
            
            print(f"\nModel Response ({len(actual)} chars):")
            print(f"{actual[:200]}..." if len(actual) > 200 else actual)
            
            print(f"\nExpected Response ({len(expected)} chars):")
            print(f"{expected[:200]}..." if len(expected) > 200 else expected)
            
            # Enhanced accuracy check with multiple methods
            similarity_score = self._calculate_similarity_score(actual, expected)
            keyword_match = self._check_keyword_similarity(actual, expected)
            structure_match = self._check_structure_similarity(actual, expected)
            
            # Combined scoring
            is_correct = (similarity_score > 0.3 or keyword_match > 0.4 or structure_match)
            
            if is_correct:
                correct += 1
            
            print(f"\nScoring:")
            print(f"  Word similarity: {similarity_score:.3f}")
            print(f"  Keyword match: {keyword_match:.3f}")
            print(f"  Structure match: {structure_match}")
            print(f"  Overall: {'CORRECT' if is_correct else 'INCORRECT'}")
            print("-" * 80)
            
            results.append({
                **result,
                'sample_id': i+1,
                'expected': expected,
                'correct': is_correct,
                'similarity_score': similarity_score,
                'keyword_match': keyword_match,
                'structure_match': structure_match
            })
        
        accuracy = correct / len(results) if results else 0
        
        summary = {
            'dataset': Path(dataset_path).name,
            'model_path': self.model_path,
            'timestamp': timestamp,
            'samples_tested': len(results),
            'accuracy': accuracy,
            'correct_answers': correct,
            'avg_similarity_score': sum(r['similarity_score'] for r in results) / len(results) if results else 0,
            'avg_keyword_match': sum(r['keyword_match'] for r in results) / len(results) if results else 0,
            'results': results
        }
        
        # Save results to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\nFinal Dataset Results:")
        print(f"Accuracy: {accuracy:.2%} ({correct}/{len(results)})")
        print(f"Average word similarity: {summary['avg_similarity_score']:.3f}")
        print(f"Average keyword match: {summary['avg_keyword_match']:.3f}")
        print(f"Results saved to: {output_file}")
        
        return summary
    
    def _check_similarity(self, actual: str, expected: str) -> bool:
        """Simple similarity check."""
        actual_words = set(actual.lower().split())
        expected_words = set(expected.lower().split())
        
        if not expected_words:
            return False
        
        overlap = len(actual_words.intersection(expected_words))
        similarity = overlap / len(expected_words)
        
        return similarity > 0.5  # 50% word overlap
    
    def _calculate_similarity_score(self, actual: str, expected: str) -> float:
        """Calculate detailed similarity score."""
        actual_words = set(actual.lower().split())
        expected_words = set(expected.lower().split())
        
        if not expected_words:
            return 0.0
        
        overlap = len(actual_words.intersection(expected_words))
        similarity = overlap / len(expected_words)
        
        return similarity
    
    def _check_keyword_similarity(self, actual: str, expected: str) -> float:
        """Check similarity based on cybersecurity keywords."""
        # Key cybersecurity terms that should match
        cybersec_keywords = [
            'threat', 'risk', 'vulnerability', 'attack', 'malware', 'phishing',
            'ransomware', 'ddos', 'sql injection', 'social engineering',
            'mitigation', 'prevention', 'detection', 'response', 'recovery',
            'firewall', 'vpn', 'encryption', 'authentication', 'monitoring',
            'incident', 'breach', 'compromise', 'exploit', 'security'
        ]
        
        actual_lower = actual.lower()
        expected_lower = expected.lower()
        
        actual_keywords = set()
        expected_keywords = set()
        
        for keyword in cybersec_keywords:
            if keyword in actual_lower:
                actual_keywords.add(keyword)
            if keyword in expected_lower:
                expected_keywords.add(keyword)
        
        if not expected_keywords:
            return 0.0
        
        overlap = len(actual_keywords.intersection(expected_keywords))
        return overlap / len(expected_keywords)
    
    def _check_structure_similarity(self, actual: str, expected: str) -> bool:
        """Check if the response has similar structure to expected."""
        # Check for structured cybersecurity response elements
        structure_elements = [
            '<THREAT_ANALYSIS>',
            '<IMPACT_ASSESSMENT>',
            '<MITIGATION_STRATEGY>',
            '<PREVENTION_MEASURES>'
        ]
        
        actual_upper = actual.upper()
        expected_upper = expected.upper()
        
        actual_elements = sum(1 for elem in structure_elements if elem in actual_upper)
        expected_elements = sum(1 for elem in structure_elements if elem in expected_upper)
        
        # If expected has structured elements, check if actual has at least some
        if expected_elements > 0:
            return actual_elements >= max(1, expected_elements // 2)
        
        # If no structured elements expected, just check for reasonable length
        return len(actual.split()) > 20


def main():
    """Main function for quick testing."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python quick_tester.py <model_path> [dataset_path]")
        print("\nExample:")
        print("  python quick_tester.py /path/to/model")
        print("  python quick_tester.py /path/to/model dataset.json")
        return
    
    model_path = sys.argv[1]
    dataset_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Initialize tester
    tester = QuickModelTester(model_path)
    
    # Run tests
    if dataset_path:
        results = tester.test_dataset(dataset_path)
        print(f"\nTesting completed! Check the output file for detailed results.")
    else:
        tester.quick_benchmark()
    
    # Start interactive mode
    tester.interactive_chat()


if __name__ == "__main__":
    main()
